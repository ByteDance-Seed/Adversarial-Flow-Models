import datetime
import time
from copy import deepcopy
from statistics import mean
from typing import Any, Dict
import torch
from kornia.augmentation import RandomAffine
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, ToTensor, ToPILImage
from tqdm import tqdm

from common.accumulator import DistributedAccumulator
from common.config import create_dataset, create_object
from common.decorators import barrier_on_entry, log_on_entry
from common.distributed import (
    convert_to_ddp,
    clip_grad_norm_,
    get_device,
    get_local_rank,
    init_torch,
)
from common.entrypoint import Entrypoint
from common.fs import download
from common.persistence import PersistenceMixin
from common.seed import set_seed, shift_seed
from common.writers import WriterMixin
from data.imagenet_resize import ImageNetResizeCrop


class ClassifierTrainer(Entrypoint, PersistenceMixin, WriterMixin):
    def entrypoint(self):
        init_torch(cudnn_benchmark=False, timeout=datetime.timedelta(seconds=3600))
        self.configure_persistence()
        self.configure_seed()
        self.configure_dataloaders()  # Start data prefetch as early as possible.
        self.configure_models()
        self.configure_optimizers()
        self.configure_accumulators()
        self.configure_writer()
        self.configure_seed()  # Reset seed again before train loop.
        self.training_loop()

    # ----------------------------- Determinism ----------------------------- #

    def configure_seed(self):
        # Get initial seed.
        self.seed = self.config.training.get("seed", None)

        # Offset by resume step.
        if self.resume:
            self.seed = shift_seed(self.seed, self.resume.step)

        # Set seed.
        set_seed(self.seed)

    # -------------------------------- Model -------------------------------- #

    def configure_models(self):
        self.configure_cls_model()
        self.configure_ema_model()
        self.configure_vae_model()
        self.configure_cls_ddp_model()

    @log_on_entry
    def configure_cls_model(self, device=get_device()):
        # Create cls model.
        self.cls = create_object(self.config.cls.model).to(device)

        # Load cls checkpoint.
        checkpoint = self.config.cls.get("checkpoint", None)
        if self.resume:
            checkpoint = self.resume.models["cls"].states.path
        if checkpoint:
            state = torch.load(download(checkpoint), map_location=device)
            self.cls.load_state_dict(state, strict=self.config.cls.get("strict", True))

        # Print model size.
        num_params = sum(p.numel() for p in self.cls.parameters() if p.requires_grad)
        self.logger.info(f"Trainable parameters: {num_params:,}")

    @log_on_entry
    def configure_ema_model(self, device=get_device()):
        # Skip if not needed.
        if not self.config.get("ema"):
            return

        # Create ema model.
        self.ema = deepcopy(self.cls).to(device)

        # Load ema checkpoint.
        checkpoint = self.config.ema.get("checkpoint", None)
        if self.resume and "ema" in self.resume.models:
            checkpoint = self.resume.models["ema"].states.path
        if checkpoint:
            state = torch.load(download(checkpoint), map_location=device)
            self.ema.load_state_dict(state, strict=self.config.ema.get("strict", True))

    @log_on_entry
    def configure_vae_model(self):
        # Create vae model.
        dtype = getattr(torch, self.config.vae.dtype)
        self.vae = create_object(self.config.vae.model)
        self.vae.requires_grad_(False).eval()
        self.vae.to(device=get_device(), dtype=dtype)

        # Load vae checkpoint.
        if self.config.vae.get("checkpoint"):
            state = torch.load(download(self.config.vae.checkpoint), map_location=get_device())
            self.vae.load_state_dict(state, strict=True)

        # Compile vae if needed.
        if self.config.vae.compile:
            self.vae.encode = torch.compile(self.vae.encode)
            self.vae.decode = torch.compile(self.vae.decode)

    # --------------------------------- DDP -------------------------------- #

    @log_on_entry
    def configure_cls_ddp_model(self):
        self.cls = convert_to_ddp(self.cls)

    # ------------------------------ Optimizer ------------------------------ #

    @log_on_entry
    def configure_optimizers(self):
        optimizer_config = OmegaConf.to_container(self.config.optimizer, resolve=True)
        optimizer_type = optimizer_config.pop("type", "AdamW")
        optimizer_cls = {
            "RMSprop": torch.optim.RMSprop,
            "AdamW": torch.optim.AdamW,
            "SGD": torch.optim.SGD,
        }[optimizer_type]

        cls_lr = optimizer_config.pop("lr")
        optimizer_checkpoint = optimizer_config.pop("checkpoint", None)

        # Create optimizer.
        self.optimizer = optimizer_cls(
            self.cls.parameters(),
            lr=cls_lr,
            **optimizer_config,
        )

        # Load state if needed.
        if self.resume and "optimizer" in self.resume.optimizers:
            optimizer_checkpoint = self.resume.optimizers["optimizer"].states.path
        if optimizer_checkpoint:
            optimizer_state = torch.load(download(optimizer_checkpoint), map_location="cpu")
            self.optimizer.load_state_dict(optimizer_state)
            del optimizer_state

        # Override lr.
        self.optimizer.param_groups[0]["lr"] = cls_lr

    # ----------------------------- Dataloaders ----------------------------- #

    def configure_dataloaders(self):
        # Create dataset.
        self.image_dataset = create_dataset(
            path=self.config.data.train.dataset.path,
            seed=self.seed,
            image_transform=Compose(
                [
                    ToTensor(),
                    RandomAffine(
                        p=0.8,
                        degrees=10,
                        translate=(0.3, 0.3),
                        scale=(1.0, 1.3),
                        resample="bilinear",
                        padding_mode="reflection",
                        keepdim=True,
                    ),
                    ToPILImage(),
                    ImageNetResizeCrop(self.config.data.train.dataset.resolution),
                    RandomHorizontalFlip(self.config.data.train.dataset.flip_prob),
                    ToTensor(),
                    Normalize(0.5, 0.5),
                ]
            ),
        )

        # Create image dataloader
        self.image_dataloader = DataLoader(
            dataset=self.image_dataset,
            batch_size=self.config.data.train.dataloader.batch_size,
            num_workers=self.config.data.train.dataloader.num_workers,
            prefetch_factor=self.config.data.train.dataloader.prefetch_factor,
            pin_memory=True,
            pin_memory_device=str(get_device()),
        )

    # ----------------------------- Accumulator ----------------------------- #

    @log_on_entry
    def configure_accumulators(self):
        self.metrics_avg_accumulator = DistributedAccumulator(mode="avg")

    # ------------------------------ Training ------------------------------- #

    @barrier_on_entry
    @log_on_entry
    def training_loop(self):
        # Set steps.
        init_step = self.resume.step if self.resume else self.config.training.get("init_step", 0)
        last_step = self.config.training.steps
        step = init_step

        # Set up progress bar.
        pbar = tqdm(
            initial=init_step,
            total=last_step,
            dynamic_ncols=True,
            disable=get_local_rank() != 0,
        )

        # Set up data iterator.
        dataloader_iter = iter(self.image_dataloader)

        # Loop over the dataloader.
        while step < last_step:

            # Get batch.
            dataloading_start_time = time.perf_counter()
            image_batch = next(dataloader_iter)
            dataloading_time = time.perf_counter() - dataloading_start_time

            # Prepare inputs.
            inputs = self.prepare_input(image_batch)

            # Train step.
            losses = self.training_step(step, **inputs)

            # Loss backward.
            losses["loss/total"].backward()

            # Optimizer step.
            stats_optimizer = self.optimizer_step()

            # Update ema.
            self.ema_step()

            # Accumulate metrics for logging.
            stats_latents = self.get_latents_statistics(inputs)
            self.metrics_avg_accumulator.add(
                **losses, **stats_latents, **stats_optimizer
            )

            # Write metrics.
            if step % self.config.writer.interval.metrics == 0:
                self.write_metrics(step)

            # Save checkpoints.
            if step % self.config.persistence.interval == 0 and step != init_step:
                self.save_checkpoint(step)

            # Update progress bar.
            pbar.set_postfix(
                {
                    "loss": losses["loss/total"].item(),
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "img": len(image_batch["image"]),
                    "t_data": dataloading_time,
                }
            )
            pbar.update()

            step += 1

    def prepare_input(
        self, image_batch: Dict[str, Any]
    ) -> Dict[str, Tensor]:
        device = get_device()
        images = image_batch["image"].to(device, non_blocking=True)
        labels = image_batch["label"].to(device, non_blocking=True)
        latents = self.vae_encode(images)
        timesteps = torch.rand([len(images)], device=device)
        noises = torch.randn_like(latents)

        return {
            "latents": latents,
            "labels": labels,
            "noises": noises,
            "timesteps": timesteps,
        }

    @staticmethod
    def interpolate(latents, noises, timesteps):
        timesteps = timesteps.view(-1, 1, 1, 1)
        return (1 - timesteps) * latents + timesteps * noises

    def training_step(
        self,
        step: int,
        *,
        latents: Tensor,
        labels: Tensor,
        noises: Tensor,
        timesteps: Tensor,
    ):
        latents_noised = self.interpolate(latents, noises, timesteps)
        prob = self.cls(latents_noised, timesteps)
        loss = torch.nn.functional.cross_entropy(prob, labels)

        return {
            "loss/total": loss
        }

    def optimizer_step(self):
        stats_optimizer = {}
        if self.config.training.gradient_clip is not None:
            stats_optimizer["grad_norm"] = clip_grad_norm_(
                self.cls,
                self.config.training.gradient_clip
            )
        self.optimizer.step()
        self.optimizer.zero_grad()
        return stats_optimizer

    def ema_step(self):
        if hasattr(self, "ema"):
            for tgt, src in zip(self.ema.parameters(), self.cls.parameters()):
                tgt.data.lerp_(src.data.to(tgt), 1 - self.config.ema.decay)

    # -------------------------------- VAE ------------------------------- #

    @torch.no_grad()
    def vae_encode(self, samples: Tensor) -> Tensor:
        dtype = getattr(torch, self.config.vae.dtype)
        scale = self.config.vae.scaling_factor
        samples = samples.to(dtype)
        samples = samples.clamp(-1, 1)
        latents = self.vae.encode(samples).latent
        latents = latents * scale
        return latents.float()

    @torch.no_grad()
    def vae_decode(self, latents: Tensor) -> Tensor:
        dtype = getattr(torch, self.config.vae.dtype)
        scale = self.config.vae.scaling_factor
        latents = latents.to(dtype)
        latents = latents / scale
        samples = self.vae.decode(latents).sample
        samples = samples.clamp(-1, 1)
        return samples.float()

    # -------------------------------- Writer ------------------------------- #

    @torch.no_grad()
    def write_metrics(self, step):
        self.writer.log_metrics(
            step=step,
            metrics={
                **self.metrics_avg_accumulator.get_and_reset(),
                **self.get_model_statistics(),
                "lr": self.optimizer.param_groups[0]["lr"],
            },
        )

    # ------------------------------ Statistics ----------------------------- #

    def get_latents_statistics(self, inputs):
        latents = inputs["latents"]
        return {
            "latents/mean": latents.mean(),
            "latents/std": latents.std(),
            "latents/seq_len": latents.numel(),
        }

    def get_model_statistics(self):
        param_mean = [p.detach().mean().item() for p in self.cls.parameters()]
        param_std = [p.detach().std().item() for p in self.cls.parameters()]
        num_buffers = sum(b.numel() for b in self.cls.buffers())
        num_params = sum(p.numel() for p in self.cls.parameters())
        num_params_trainable = sum(p.numel() for p in self.cls.parameters() if p.requires_grad)
        return {
            "model/num_buffers": num_buffers,
            "model/num_params": num_params,
            "model/num_params_trainable": num_params_trainable,
            "model/param_mean_avg": mean(param_mean),
            "model/param_mean_min": min(param_mean),
            "model/param_mean_max": max(param_mean),
            "model/param_std_avg": mean(param_std),
            "model/param_std_min": min(param_std),
            "model/param_std_max": max(param_std),
        }

    @log_on_entry
    def save_checkpoint(self, step):
        self.persistence.save_model(
            step=step,
            name="cls",
            config=self.config.cls.model,
            states=self.cls.module.state_dict(),
            blocking=True,
        )
        if hasattr(self, "ema"):
            self.persistence.save_model(
                step=step,
                name="ema",
                config=self.config.cls.model,
                states=self.ema.state_dict(),
                blocking=True,
            )
        self.persistence.save_optimizer(
            step=step,
            name="optimizer",
            states=self.optimizer.state_dict(),
            blocking=True,
        )
