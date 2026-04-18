import datetime
import time
from copy import deepcopy
from statistics import mean
from typing import Any, Dict
import torch
from omegaconf import OmegaConf
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, ToTensor
from tqdm import tqdm

from common.accumulator import DistributedAccumulator
from common.config import create_dataset, create_object
from common.decorators import barrier_on_entry, global_rank_zero_only, log_on_entry
from common.distributed import (
    convert_to_ddp,
    clip_grad_norm_,
    get_device,
    get_global_rank,
    get_local_rank,
    get_world_size,
    init_torch,
)
from common.entrypoint import Entrypoint
from common.fs import download
from common.metrics import ModifiedMetricCollection
from common.partition import partition_by_groups
from common.persistence import PersistenceMixin
from common.seed import set_seed, shift_seed
from common.writers import WriterMixin
from data.imagenet_resize import ImageNetResizeCrop
from models.cafm.jvp.discriminator import DiscriminatorJVP

torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

class ContinuousAdversarialFlowTrainer(Entrypoint, PersistenceMixin, WriterMixin):
    def entrypoint(self):
        init_torch(cudnn_benchmark=True, timeout=datetime.timedelta(seconds=3600))
        self.configure_persistence()
        self.configure_seed()
        self.configure_dataloaders()
        self.configure_models()
        self.configure_optimizers()
        self.configure_accumulators()
        self.configure_evaluators()
        self.configure_writer()
        self.configure_seed()
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
        self.configure_gen_model()
        self.configure_dis_model()
        self.configure_ema_model()
        self.configure_vae_model()
        self.configure_gen_ddp_model()
        self.configure_dis_ddp_model()

    @log_on_entry
    def configure_gen_model(self, device=get_device()):
        # Create gen model.
        self.gen = create_object(self.config.gen.model).to(device)

        # Load gen checkpoint.
        checkpoint = self.config.gen.get("checkpoint", None)
        if self.resume:
            checkpoint = self.resume.models["gen"].states.path
        if checkpoint:
            state = torch.load(download(checkpoint), map_location=device)
            self.gen.load_state_dict(state, strict=self.config.gen.get("strict", True))

        # Print model size.
        num_params = sum(p.numel() for p in self.gen.parameters() if p.requires_grad)
        self.logger.info(f"Gen trainable parameters: {num_params:,}")

    @log_on_entry
    def configure_dis_model(self, device=get_device()):
        # Create discriminator model.
        self.dis = create_object(self.config.dis.model).to(device)

        # Load discriminator checkpoint.
        checkpoint = self.config.dis.get("checkpoint", None)
        if self.resume and "dis" in self.resume.models:
            checkpoint = self.resume.models["dis"].states.path
        if checkpoint:
            state = torch.load(download(checkpoint), map_location=device)
            self.dis.load_state_dict(state, strict=self.config.dis.get("strict", True))

    @log_on_entry
    def configure_ema_model(self, device=get_device()):
        # Skip if not needed.
        if not self.config.get("ema"):
            return

        # Create ema model.
        self.ema = deepcopy(self.gen).to(device)

        # Load ema checkpoint.
        checkpoint = self.config.ema.get("checkpoint", None)
        if self.resume and "ema" in self.resume.models:
            checkpoint = self.resume.models["ema"].states.path
        if checkpoint:
            state = torch.load(download(checkpoint), map_location=device)
            self.ema.load_state_dict(state, strict=self.config.ema.get("strict", True))

    @log_on_entry
    def configure_vae_model(self):
        # Skip configure vae if not needed
        if self.config.vae is None:
            self.vae = None
            return

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
    def configure_gen_ddp_model(self):
        if self.config.gen.get("compile", True):
            self.gen = torch.compile(self.gen)
        self.gen = convert_to_ddp(self.gen)

    @log_on_entry
    def configure_dis_ddp_model(self):
        if self.config.dis.get("compile", True):
            self.dis = torch.compile(self.dis)
        self.dis = DiscriminatorJVP(self.dis)
        self.dis = convert_to_ddp(self.dis)

        self.dis_trainable_params = [
            param for param in self.dis.parameters() if param.requires_grad
        ]

    # ------------------------------ Optimizer ------------------------------ #

    @log_on_entry
    def configure_optimizers(self):
        optimizer_config = OmegaConf.to_container(self.config.optimizer, resolve=True)

        gen_lr = optimizer_config.pop("gen_lr")
        dis_lr = optimizer_config.pop("dis_lr", gen_lr)
        weight_decay = optimizer_config.pop("weight_decay")
        betas = optimizer_config.pop("betas", [0.0, 0.95])
        gen_optimizer_checkpoint = optimizer_config.pop("gen_checkpoint", None)
        dis_optimizer_checkpoint = optimizer_config.pop("dis_checkpoint", None)

        self.gen_optimizer = AdamW(params=self.gen.parameters(), lr=gen_lr, weight_decay=weight_decay, betas=betas, fused=True)
        self.dis_optimizer = AdamW(params=self.dis.parameters(), lr=dis_lr, weight_decay=weight_decay, betas=betas, fused=True)

        # Load state if needed.
        if self.resume and "gen_optimizer" in self.resume.optimizers:
            gen_optimizer_checkpoint = self.resume.optimizers["gen_optimizer"].states.path
        if gen_optimizer_checkpoint:
            gen_optimizer_state = torch.load(download(gen_optimizer_checkpoint), map_location="cpu")
            self.gen_optimizer.load_state_dict(gen_optimizer_state)
            del gen_optimizer_state

        if self.resume and "dis_optimizer" in self.resume.optimizers:
            dis_optimizer_checkpoint = self.resume.optimizers["dis_optimizer"].states.path
        if dis_optimizer_checkpoint:
            dis_optimizer_state = torch.load(download(dis_optimizer_checkpoint), map_location="cpu")
            self.dis_optimizer.load_state_dict(dis_optimizer_state)
            del dis_optimizer_state

        # Override lr if resumed from old ckpt.
        self.gen_optimizer.param_groups[0]["lr"] = gen_lr
        self.dis_optimizer.param_groups[0]["lr"] = dis_lr

    # ----------------------------- Dataloaders ----------------------------- #

    def configure_dataloaders(self):
        # Create dataset.
        self.image_dataset = create_dataset(
            path=self.config.data.train.dataset.path,
            seed=self.seed,
            image_transform=Compose(
                [
                    ImageNetResizeCrop(self.config.data.train.dataset.resolution),
                    RandomHorizontalFlip(self.config.data.train.dataset.flip_prob),
                    ToTensor(),
                    Normalize(0.5, 0.5),
                ]
            ),
        )

        # Create dataloader
        self.gen_dataloader = DataLoader(
            dataset=self.image_dataset,
            batch_size=self.config.data.train.dataloader.batch_size,
            num_workers=self.config.data.train.dataloader.num_workers,
            prefetch_factor=self.config.data.train.dataloader.prefetch_factor,
            pin_memory=True,
            pin_memory_device=str(get_device()),
        )

        self.dis_dataloader = DataLoader(
            dataset=self.image_dataset,
            batch_size=self.config.data.train.dataloader.batch_size,
            num_workers=self.config.data.train.dataloader.num_workers,
            prefetch_factor=self.config.data.train.dataloader.prefetch_factor,
            pin_memory=True,
            pin_memory_device=str(get_device()),
        )

        # Set up data iterator.
        self.gen_dataloader_iter = iter(self.gen_dataloader)
        self.dis_dataloader_iter = iter(self.dis_dataloader)

    # ----------------------------- Accumulator ----------------------------- #

    @log_on_entry
    def configure_accumulators(self):
        self.metrics_avg_accumulator = DistributedAccumulator(mode="avg")

    # ------------------------------ Evaluator ------------------------------ #

    @log_on_entry
    def configure_evaluators(self):
        if self.config.get("evaluation") is not None:
            metrics = {}
            for key, metric in self.config.evaluation.metrics.items():
                metrics[key] = create_object(metric).to(get_device())
            self.evaluator = ModifiedMetricCollection(metrics)

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

        # Loop over the dataloader.
        while step < last_step:

            # Get batch.
            dataloading_start_time = time.perf_counter()
            image_batch = self.get_data(step)
            dataloading_time = time.perf_counter() - dataloading_start_time

            # Prepare inputs.
            inputs = self.prepare_input(image_batch)

            # Train step.
            losses, results = self.training_step(step, **inputs)

            # Loss backward.
            accu = self.config.training.get("accumulation", 1)
            losses["loss/total"].div(accu).backward()

            # Optimizer step.
            stats_optimizer = self.optimizer_step(step)

            # Update ema.
            self.ema_step(step)

            # Accumulate metrics for logging.
            stats_latents = self.get_latents_statistics(inputs)
            self.metrics_avg_accumulator.add(
                **losses, **stats_latents, **stats_optimizer
            )

            # Write metrics.
            if step % self.config.writer.interval.metrics == 0:
                self.write_metrics(step)

            # Write visuals.
            if (
                self.config.writer.interval.visuals > 0
                and step % self.config.writer.interval.visuals == 0
            ):
                self.write_visuals(step, image_batch, inputs, results)

            # Save checkpoints.
            if step % self.config.persistence.interval == 0 and step != init_step:
                self.save_checkpoint(step)

            # Validation.
            if self.config.validation.enabled and step % self.config.validation.interval == 0:
                self.validation(step, image_batch, inputs, results, use_ema=False)
                self.validation(step, image_batch, inputs, results, use_ema=True)

            # Evaluation.
            if (
                self.config.get("evaluation")
                and (step != init_step or self.config.evaluation.get("on_init_step", False))
                and step % self.config.evaluation.interval == 0
            ):
                self.evaluate(step, use_ema=False)
                self.evaluate(step, use_ema=True)

            # Update progress bar.
            pbar.set_postfix(
                {
                    "loss": losses["loss/total"].item(),
                    "img": len(image_batch["label"]),
                    "t_data": dataloading_time,
                }
            )
            pbar.update()

            step += 1

    def get_data(self, step):
        if self.is_dis_step(step):
            return next(self.dis_dataloader_iter)
        else:
            return next(self.gen_dataloader_iter)

    def prepare_input(self, image_batch: Dict[str, Any]) -> Dict[str, Tensor]:
        device = get_device()
        
        # Get latents from dataset.
        if "latent" in image_batch.keys():
            # Offline dataset. Latent precomputed.
            latents = image_batch["latent"].to(device, non_blocking=True)
        else:
            # Online dataset. Encode images to latents.
            images = image_batch["image"].to(device, non_blocking=True)
            latents = self.vae_encode(images)

        # Get class labels.
        labels = image_batch["label"].to(device, non_blocking=True)

        # Class dropout if needed.
        dropmask = torch.rand(len(labels), device=device) >= self.config.gen.get("dropout", 0.0)
        labels = torch.where(dropmask, labels, 1000)

        # Sample timesteps.
        timesteps_mode = self.config.training.get("timesteps", "uniform")
        if timesteps_mode == "uniform":
            timesteps = torch.rand([len(labels)], device=device)
        elif timesteps_mode == "logisticnormal":
            timesteps = torch.randn([len(labels)], device=device).sigmoid()
        elif timesteps_mode == "logisticnormal_jit":
            # jit table 9. shift +0.8 because our training code uses x0=image, x1=noise.
            timesteps = torch.randn([len(labels)], device=device).mul(0.8).add(0.8).sigmoid()
        
        # Sample noises.
        noises = torch.randn_like(latents)

        return {
            "latents": latents,
            "labels": labels,
            "noises": noises,
            "timesteps": timesteps,
        }

    def is_dis_step(self, step: int):
        accu = self.config.training.get("accumulation", 1)
        warm = self.config.adversarial.get("warmup", 1)
        N = self.config.adversarial.get("N", 1)
        if step // accu <= warm:
            return True
        return step // accu % (1 + N) < N

    def is_update_step(self, step: int):
        accu = self.config.training.get("accumulation", 1)
        return step % accu == accu - 1

    def interpolate(self, latents, noises, timesteps):
        timesteps = timesteps.reshape(-1, 1, 1, 1)
        return (1 - timesteps) * latents + (timesteps) * noises

    def velocity(self, latents, noises, timesteps):
        return noises - latents

    def training_step(
        self,
        step: int,
        *,
        latents: Tensor,
        labels: Tensor,
        noises: Tensor,
        timesteps: Tensor,
    ):
        dis_step = self.is_dis_step(step)

        latents_noised = self.interpolate(latents, noises, timesteps)

        torch.set_grad_enabled(not dis_step)
        velocity_real = self.velocity(latents, noises, timesteps)
        velocity_pred = self.gen(latents_noised, labels, timesteps)
        torch.set_grad_enabled(True)

        for param in self.dis_trainable_params:
            param.requires_grad_(dis_step)

        if dis_step:
            cp_scale = self.config.adversarial.get("cp_scale", 0.001)

            # Internally it uses vmap to fuse jvp computation of multiple tangents.
            tangent_x = torch.stack([velocity_real, velocity_pred])
            tangent_t = torch.stack([torch.ones_like(timesteps), torch.ones_like(timesteps)])

            out, out_jvp = self.dis(
                x=latents_noised,
                t=timesteps,
                y=labels,
                dx=tangent_x,
                dt=tangent_t,
            )

            logits_real, logits_fake = out_jvp.chunk(2, 0)

            dis_adv_real_loss = logits_real.sub(1).square().mean()
            dis_adv_fake_loss = logits_fake.add(1).square().mean()
            dis_adv_loss = dis_adv_real_loss + dis_adv_fake_loss
            dis_cp_loss = out.square().mean().mul(cp_scale)

            loss_dict = {
                "loss/total": (dis_adv_loss + dis_cp_loss),
                "loss/dis": dis_adv_loss,
                "loss/cp": dis_cp_loss,
                "logits/real": logits_real.detach().mean(),
                "logits/fake": logits_fake.detach().mean(),
                "logits/out": out.detach().mean(),
            }
        else:
            ot_scale = self.config.adversarial.get("ot_scale", 0.0)

            _, logits_fake = self.dis(
                x=latents_noised,
                t=timesteps,
                y=labels,
                dx=velocity_pred,
                dt=torch.ones_like(timesteps),
            )

            gen_adv_loss = logits_fake.sub(1).square().mean()
            gen_ot_loss = velocity_pred.square().mean().mul(ot_scale)

            loss_dict = {
                "loss/total": (gen_adv_loss + gen_ot_loss),
                "loss/gen": gen_adv_loss,
                "loss/ot": gen_ot_loss,
                "logits/fake": logits_fake.detach().mean(),
            }

        visual_dict = {
            "latents_pred": latents_noised - timesteps.view(-1, 1, 1, 1) * velocity_pred,
        }

        return loss_dict, visual_dict


    def optimizer_step(self, step):
        stats_optimizer = {}

        if not self.is_update_step(step):
            return stats_optimizer

        if self.is_dis_step(step):
            if self.config.training.get("dis_gradient_clip", None) is not None:
                stats_optimizer["dis_grad_norm"] = clip_grad_norm_(
                    self.dis,
                    self.config.training.dis_gradient_clip
                )

            self.dis_optimizer.step()
        else:
            if self.config.training.gradient_clip is not None:
                stats_optimizer["gen_grad_norm"] = clip_grad_norm_(
                    self.gen,
                    self.config.training.gradient_clip
                )

            self.gen_optimizer.step()

        self.dis.zero_grad()
        self.gen.zero_grad()

        return stats_optimizer

    def ema_step(self, step):
        if not self.is_update_step(step):
            return

        if self.is_dis_step(step):
            return

        if hasattr(self, "ema"):
            for tgt, src in zip(self.ema.parameters(), self.gen.parameters()):
                tgt.data.lerp_(src.data.to(tgt), 1 - self.config.ema.decay)

    # -------------------------------- VAE ------------------------------- #

    @torch.no_grad()
    def vae_encode(self, samples: Tensor) -> Tensor:
        if self.vae is None:
            return samples.clamp(-1, 1)
        dtype = getattr(torch, self.config.vae.dtype)
        scale = self.config.vae.scaling_factor
        samples = samples.to(dtype)
        samples = samples.clamp(-1, 1)
        latents = self.vae.encode(samples).latent
        latents = latents * scale
        return latents.float()

    @torch.no_grad()
    def vae_decode(self, latents: Tensor) -> Tensor:
        if self.vae is None:
            return latents.clamp(-1, 1)
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
                **self.get_gen_model_statistics(),
                **self.get_dis_model_statistics(),
                "gen_lr": self.gen_optimizer.param_groups[0]["lr"],
                "dis_lr": self.dis_optimizer.param_groups[0]["lr"],
            },
        )

    @torch.no_grad()
    @global_rank_zero_only
    @log_on_entry
    def write_visuals(self, step, batch, inputs, results):
        max_batch_size = self.config.writer.max_batch_size
        latents = inputs["latents"][:max_batch_size]
        latents_pred = results["latents_pred"][:max_batch_size]
        images = self.vae_decode(latents)
        images_pred = self.vae_decode(latents_pred)
        self.writer.log_images(
            step=step,
            images={
                "image/data": images,
                "image/pred": images_pred,
            },
            captions={
                "image/data": list(map(str, inputs["labels"].tolist()))[:max_batch_size],
                "image/pred": list(map(str, inputs["timesteps"].tolist()))[:max_batch_size],
            },
            value_range=(-1, 1),
        )

    # ------------------------------ Statistics ----------------------------- #

    def get_latents_statistics(self, inputs):
        latents = inputs["latents"]
        return {
            "latents/mean": latents.mean(),
            "latents/std": latents.std(),
            "latents/seq_len": latents.numel(),
        }

    def get_gen_model_statistics(self):
        param_mean = [p.detach().mean().item() for p in self.gen.parameters()]
        param_std = [p.detach().std().item() for p in self.gen.parameters()]
        num_buffers = sum(b.numel() for b in self.gen.buffers())
        num_params = sum(p.numel() for p in self.gen.parameters())
        num_params_trainable = sum(p.numel() for p in self.gen.parameters() if p.requires_grad)
        return {
            "model_gen/num_buffers": num_buffers,
            "model_gen/num_params": num_params,
            "model_gen/num_params_trainable": num_params_trainable,
            "model_gen/param_mean_avg": mean(param_mean),
            "model_gen/param_mean_min": min(param_mean),
            "model_gen/param_mean_max": max(param_mean),
            "model_gen/param_std_avg": mean(param_std),
            "model_gen/param_std_min": min(param_std),
            "model_gen/param_std_max": max(param_std),
        }

    def get_dis_model_statistics(self):
        param_mean = [p.detach().mean().item() for p in self.dis.parameters()]
        param_std = [p.detach().std().item() for p in self.dis.parameters()]
        num_buffers = sum(b.numel() for b in self.dis.buffers())
        num_params = sum(p.numel() for p in self.dis.parameters())
        num_params_trainable = sum(p.numel() for p in self.dis.parameters() if p.requires_grad)
        return {
            "model_dis/num_buffers": num_buffers,
            "model_dis/num_params": num_params,
            "model_dis/num_params_trainable": num_params_trainable,
            "model_dis/param_mean_avg": mean(param_mean),
            "model_dis/param_mean_min": min(param_mean),
            "model_dis/param_mean_max": max(param_mean),
            "model_dis/param_std_avg": mean(param_std),
            "model_dis/param_std_min": min(param_std),
            "model_dis/param_std_max": max(param_std),
        }

    # ------------------------------ Inference ------------------------------ #

    @torch.no_grad()
    def validation(self, step, image_batch, inputs, results, use_ema=False):
        noises = inputs["noises"]

        B = self.config.validation.max_batch_size
        _, C, H, W = noises.shape

        noises = torch.randn(
            size=[B, C, H, W],
            device=noises.device,
            dtype=noises.dtype,
            generator=torch.Generator("cuda").manual_seed(0),
        )
        labels = torch.arange(0, len(noises), device=noises.device)

        samples = self.inference(
            noises=noises,
            labels=labels,
            use_ema=use_ema,
        )
        model_type = "_ema" if use_ema else ""

        self.writer.log_images(
            step=step,
            images={
                f"image/validation{model_type}": samples,
            },
            value_range=(-1, 1),
        )

    @torch.no_grad()
    def inference(
        self,
        noises: Tensor,
        labels: Tensor,
        use_ema: bool = False,
    ) -> Tensor:
        steps = self.config.validation.get("steps", 250)
        sampler = self.config.validation.get("sampler", "heun")

        device = noises.device
        model = self.ema if use_ema else self.gen
        model.eval()
        
        if sampler == "euler":
            timesteps = torch.linspace(1.0, 0.0, steps+1, device=device)
            latents = noises
            for timesteps_src, timesteps_tgt in zip(timesteps[:-1], timesteps[1:]):
                timesteps_src = timesteps_src.repeat(len(noises))
                timesteps_tgt = timesteps_tgt.repeat(len(noises))
                outputs = model(latents, labels, timesteps_src)
                latents = latents - (timesteps_src - timesteps_tgt).view(-1, 1, 1, 1) * outputs
        elif sampler == "heun":
            timesteps = torch.linspace(1.0, 0.0, (steps // 2)+1, device=device)
            latents = noises
            for timesteps_src, timesteps_tgt in zip(timesteps[:-1], timesteps[1:]):
                timesteps_src = timesteps_src.repeat(len(noises))
                timesteps_tgt = timesteps_tgt.repeat(len(noises))
                outputs1 = model(latents, labels, timesteps_src)
                latents_next = latents - (timesteps_src - timesteps_tgt).view(-1, 1, 1, 1) * outputs1
                outputs2 = model(latents_next, labels, timesteps_tgt)
                latents = latents - (timesteps_src - timesteps_tgt).view(-1, 1, 1, 1) * 0.5 * (outputs1 + outputs2)
        else:
            raise NotImplementedError(f"Unknown sampler: {sampler}")

        model.train()
        samples = self.vae_decode(latents)
        return samples

    # ----------------------------- Evaluation ------------------------------ #

    @torch.no_grad()
    def evaluate(self, step, use_ema):
        device = get_device()

        resolution = self.config.data.train.dataset.resolution

        labels = torch.arange(0, 1000).repeat_interleave(50).tolist() # 50000 samples
        labels = partition_by_groups(labels, get_world_size())[get_global_rank()]

        for labels in tqdm(
            iterable=partition_by_groups(labels, int(50000 / get_world_size() / 32)),
            disable=get_local_rank() != 0,
            desc="Evaluation",
            dynamic_ncols=True,
        ):
            labels = torch.tensor(labels, device=device, dtype=torch.long)

            if self.vae is not None:
                noises = torch.randn(
                    size=[len(labels), 4, resolution // 8, resolution // 8],
                    device=device,
                )
            else:
                noises = torch.randn(
                    size=[len(labels), 3, resolution, resolution],
                    device=device,
                )

            samples = self.inference(
                noises=noises,
                labels=labels,
                use_ema=use_ema,
            )
            samples = samples * 0.5 + 0.5

            self.evaluator.update(preds=samples, target=None)

        model_type = "_ema" if use_ema else ""
        results = self.evaluator.compute_and_reset()
        results = {f"eval/{key}{model_type}": value.item() for key, value in results.items()}

        self.writer.log_metrics(step=step, metrics=results)
        self.logger.info(f"Evaluation Results: {results}")

    # ----------------------------- Persistence ----------------------------- #

    @log_on_entry
    def save_checkpoint(self, step):
        self.persistence.save_model(
            step=step,
            name="gen",
            config=self.config.gen.model,
            states=(
               self.gen.module._orig_mod.state_dict()
               if self.config.gen.get("compile", True)
               else self.gen.module.state_dict()
            ),
            blocking=True,
        )
        self.persistence.save_model(
            step=step,
            name="dis",
            config=self.config.dis.model,
            states=(
                self.dis.module.dis._orig_mod.state_dict()
                if self.config.dis.get("compile", True)
                else self.dis.module.dis.state_dict()
            ),
            blocking=True,
        )
        if hasattr(self, "ema"):
            self.persistence.save_model(
                step=step,
                name="ema",
                config=self.config.gen.model,
                states=self.ema.state_dict(),
                blocking=True,
            )

        self.persistence.save_optimizer(
            step=step,
            name="gen_optimizer",
            states=self.gen_optimizer.state_dict(),
            blocking=True,
        )
        self.persistence.save_optimizer(
            step=step,
            name="dis_optimizer",
            states=self.dis_optimizer.state_dict(),
            blocking=True,
        )