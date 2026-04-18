import datetime
import time
from copy import deepcopy
from statistics import mean
from typing import Any, Dict
import torch
from kornia.augmentation import RandomErasing, RandomTranslate
from omegaconf import OmegaConf
from torch import Tensor
from torch.nn import functional as F
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
from common.gradcapture import GradientCapture
from common.metrics import ModifiedMetricCollection
from common.partition import partition_by_groups
from common.persistence import PersistenceMixin
from common.schedules import apply_lr
from common.schedules import create_schedule_from_config as create_schedule
from common.seed import set_seed, shift_seed
from common.writers import WriterMixin
from data.imagenet_resize import ImageNetResizeCrop
from grad_norm import GradientNormalization


class AdversarialFlowTrainer(Entrypoint, PersistenceMixin, WriterMixin):
    def entrypoint(self):
        init_torch(cudnn_benchmark=False, timeout=datetime.timedelta(seconds=3600))
        self.configure_persistence()
        self.configure_seed()
        self.configure_dataloaders()
        self.configure_models()
        self.configure_optimizers()
        self.configure_accumulators()
        self.configure_evaluators()
        self.configure_gradient_capture()
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
        self.configure_cls_model()
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

    @log_on_entry
    def configure_cls_model(self, device=get_device()):
        if self.config.get("cls") is None:
            self.cls = None
            return

        # Create cls model.
        self.cls = create_object(self.config.cls.model).to(device).requires_grad_(False).eval()

        # Load cls checkpoint.
        checkpoint = self.config.cls.checkpoint
        state = torch.load(download(checkpoint), map_location=device)
        self.cls.load_state_dict(state)

        # Compile vae if needed.
        if self.config.cls.compile:
            self.cls = torch.compile(self.cls)

    # --------------------------------- DDP -------------------------------- #

    @log_on_entry
    def configure_gen_ddp_model(self):
        self.gen = convert_to_ddp(self.gen)

    @log_on_entry
    def configure_dis_ddp_model(self):
        self.dis = convert_to_ddp(self.dis)
        self.dis_trainable_params = [
            param for param in self.dis.parameters() if param.requires_grad
        ]

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

        gen_lr = optimizer_config.pop("gen_lr")
        dis_lr = optimizer_config.pop("dis_lr", gen_lr)
        gen_optimizer_checkpoint = optimizer_config.pop("gen_checkpoint", None)
        dis_optimizer_checkpoint = optimizer_config.pop("dis_checkpoint", None)

        # Create optimizer.
        self.gen_optimizer = optimizer_cls(
            self.gen.parameters(),
            lr=gen_lr,
            **optimizer_config,
        )

        self.dis_optimizer = optimizer_cls(
            self.dis.parameters(),
            lr=dis_lr,
            **optimizer_config,
        )

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

        # Override lr.
        self.gen_optimizer.param_groups[0]["lr"] = gen_lr
        self.dis_optimizer.param_groups[0]["lr"] = dis_lr

        if optimizer_config.get("betas", None) is not None:
            betas = optimizer_config["betas"]
            self.gen_optimizer.param_groups[0]["betas"] = betas
            self.dis_optimizer.param_groups[0]["betas"] = betas

        if optimizer_config.get("eps", None) is not None:
            eps = optimizer_config["eps"]
            self.gen_optimizer.param_groups[0]["eps"] = eps
            self.dis_optimizer.param_groups[0]["eps"] = eps

        # For compatibility with the base class statistic logging.
        self.optimizer = self.gen_optimizer

        # Create lr schedule.
        self.gen_lr_schedule = None
        self.dis_lr_schedule = None
        if self.config.get("gen_lr_schedule"):
            self.gen_lr_schedule = create_schedule(self.config.gen_lr_schedule)
        if self.config.get("dis_lr_schedule"):
            self.dis_lr_schedule = create_schedule(self.config.dis_lr_schedule)

        # Create loss schedule.
        if self.config.get("gp_scale_schedule"):
            self.gp_scale_schedule = create_schedule(self.config.gp_scale_schedule)
        if self.config.get("ot_scale_schedule"):
            self.ot_scale_schedule = create_schedule(self.config.ot_scale_schedule)
        if self.config.get("aug_p_schedule"):
            self.aug_p_schedule = create_schedule(self.config.aug_p_schedule)

        self.grad_norm = torch.nn.Identity()
        if self.config.adversarial.grad_norm:
            self.grad_norm = GradientNormalization().to(get_device())
            if self.resume and "grad_norm" in self.resume.optimizers:
                grad_norm_checkpoint = self.resume.optimizers["grad_norm"].states.path
                grad_norm_state = torch.load(download(grad_norm_checkpoint), map_location="cpu")
                self.grad_norm.load_state_dict(grad_norm_state, strict=True)

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

    def configure_gradient_capture(self):
        self.gen_dis_grad_capture = GradientCapture()
        self.gen_ot_grad_capture = GradientCapture()

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
            losses["loss/total"].backward()

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
                self.validation(step, image_batch, inputs, results)

            # Evaluation.
            if (
                step != 0
                and self.config.get("evaluation")
                and step % self.config.evaluation.interval == 0
            ):
                self.evaluate(step)

                if self.config.ema.get("autoreload", False):
                    for dit_p, ema_p in zip(self.dit.parameters(), self.ema.parameters()):
                        dit_p.data.copy_(ema_p.data)

                if self.config.dis.get("autoreload", False):
                    checkpoint = self.config.dis.checkpoint
                    state = torch.load(download(checkpoint), map_location=get_device())
                    self.dis.module.load_state_dict(state, strict=self.config.dis.get("strict", True))
                    del state

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

    def get_data(self, step):
        if self.is_dis_step(step):
            return next(self.dis_dataloader_iter)
        else:
            return next(self.gen_dataloader_iter)

    def prepare_input(
        self, image_batch: Dict[str, Any]
    ) -> Dict[str, Tensor]:
        device = get_device()
        images = image_batch["image"].to(device, non_blocking=True)
        labels = image_batch["label"].to(device, non_blocking=True)
        latents = self.vae_encode(images)

        steps = self.config.adversarial.get("steps", 1)
        if steps == "any":
            timesteps_src = torch.rand([len(images)], device=device)
            timesteps_tgt = torch.rand([len(images)], device=device) * timesteps_src
        elif isinstance(steps, int):
            timesteps_src = torch.arange(1.0, 0.0, -1/steps, device=device)
            timesteps_src = timesteps_src[torch.randint(0, steps, [len(images)], device=device)]
            timesteps_tgt = timesteps_src - 1/steps
        else:
            raise NotImplementedError

        timesteps_cg = None
        if self.cls is not None:
            cg_flow = self.config.adversarial.get("cg_flow", True)
            cg_timestep_lo = self.config.adversarial.get("cg_timestep_lo", 0)
            cg_timestep_hi = self.config.adversarial.get("cg_timestep_hi", 1)
            if cg_flow:
                timesteps_cg = torch.rand([len(labels)], device=device) * (cg_timestep_lo - cg_timestep_hi) + cg_timestep_hi
            else:
                timesteps_cg = timesteps_tgt.clone()

        return {
            "latents": latents,
            "labels": labels,
            "noises_src": torch.randn_like(latents),
            "noises_tgt": torch.randn_like(latents),
            "noises_r1": torch.randn_like(latents),
            "noises_r2": torch.randn_like(latents),
            "noises_cg": torch.randn_like(latents),
            "timesteps_src": timesteps_src,
            "timesteps_tgt": timesteps_tgt,
            "timesteps_cg": timesteps_cg,
        }

    def is_dis_step(self, step: int):
        accu = self.config.training.get("accumulation", 1)
        ttur = self.config.adversarial.get("ttur", 1)
        return step // accu % (1 + ttur) < ttur

    def is_update_step(self, step: int):
        accu = self.config.training.get("accumulation", 1)
        return step % accu == accu - 1

    def get_gp_scale(self, step: int):
        if hasattr(self, "gp_scale_schedule"):
            return self.gp_scale_schedule[step]
        return self.config.adversarial.get("gp_scale", 0.0)

    def get_ot_scale(self, step: int):
        if hasattr(self, "ot_scale_schedule"):
            return self.ot_scale_schedule[step]
        return self.config.adversarial.get("ot_scale", 0.0)
    
    def get_aug_p(self, step: int):
        if hasattr(self, "augment_p_schedule"):
            return self.aug_p_schedule[step]
        return self.config.adversarial.get("augment_p", 0.0)

    def augment(self, step: int, real: Tensor, fake: Tensor):
        p = self.get_aug_p(step)
        if p > 0:
            augment = torch.nn.Sequential(
                RandomTranslate(p=p, translate_x=(0, 0.3), translate_y=(0, 0.3), resample="NEAREST"),
                RandomErasing(p=p, scale=(0.1, 0.5)),
                RandomErasing(p=p, scale=(0.1, 0.5)),
                RandomErasing(p=p, scale=(0.1, 0.5)),
            )
            # HACK: We channel concat real and fake to ensure the same augmentation is applied.
            # This only work when the random augmentation does not independently modify the channels.
            real, fake = augment(torch.cat([real, fake], dim=1)).chunk(2, dim=1)
        return real, fake

    def interpolate(self, latents, noises, timesteps):
        timesteps = timesteps.reshape(-1, 1, 1, 1)
        return (1 - timesteps) * latents + (timesteps) * noises

    def training_step(
        self,
        step: int,
        *,
        latents: Tensor,
        labels: Tensor,
        noises_src: Tensor,
        noises_tgt: Tensor,
        noises_r1: Tensor,
        noises_r2: Tensor,
        noises_cg: Tensor,
        timesteps_src: Tensor,
        timesteps_tgt: Tensor,
        timesteps_cg: Tensor,
    ):
        dis_step = self.is_dis_step(step)
        accu = self.config.training.get("accumulation", 1)

        latents_src = self.interpolate(latents, noises_src, timesteps_src)
        latents_tgt = self.interpolate(latents, noises_tgt, timesteps_tgt)

        torch.set_grad_enabled(not dis_step)
        latents_tgt_pred = self.gen(latents_src, labels, timesteps_src, timesteps_tgt)
        torch.set_grad_enabled(True)

        if self.config.gen.get("pred_type") == "v":
            latents_tgt_pred = latents_src + (timesteps_tgt - timesteps_src).view(-1, 1, 1, 1) * latents_tgt_pred

        latents_tgt_real_aug, latents_tgt_pred_aug = self.augment(step, latents_tgt, latents_tgt_pred)

        for param in self.dis_trainable_params:
            param.requires_grad_(dis_step)

        weighting = (timesteps_src - timesteps_tgt).abs().clamp_min(0.001)

        if dis_step:
            bsz = len(latents)
            gp_scale = self.get_gp_scale(step)
            gp_bsz_ratio = self.config.adversarial.gp_bsz_ratio
            gp_bsz = max(round(bsz * gp_bsz_ratio), 1)
            gp_eps = self.config.adversarial.gp_eps
            cp_scale = self.config.adversarial.cp_scale

            latents_tgt_real_gp = latents_tgt_real_aug[:gp_bsz] + gp_eps * noises_r1[:gp_bsz]
            latents_tgt_pred_gp = latents_tgt_pred_aug[:gp_bsz] + gp_eps * noises_r2[:gp_bsz]

            logits_real, logits_fake, logits_real_gp, logits_fake_gp = self.dis(
                torch.cat([latents_tgt_real_aug, latents_tgt_pred_aug, latents_tgt_real_gp, latents_tgt_pred_gp]),
                torch.cat([labels, labels, labels[:gp_bsz], labels[:gp_bsz]]),
                torch.cat([timesteps_tgt, timesteps_tgt, timesteps_tgt[:gp_bsz], timesteps_tgt[:gp_bsz]]),
            ).split([bsz, bsz, gp_bsz, gp_bsz])

            dis_loss_adv = F.softplus(-(logits_real - logits_fake)).mean()
            dis_loss_r1 = (logits_real_gp - logits_real[:gp_bsz]).square().mul(gp_scale * weighting[:gp_bsz] / gp_eps ** 2).mean()
            dis_loss_r2 = (logits_fake_gp - logits_fake[:gp_bsz]).square().mul(gp_scale * weighting[:gp_bsz] / gp_eps ** 2).mean()
            dis_loss_cp = (logits_real + logits_fake).square().mul(cp_scale).mean()

            loss_dict = {
                "loss/total": (dis_loss_adv + dis_loss_r1 + dis_loss_r2 + dis_loss_cp) / accu,
                "loss/dis": dis_loss_adv,
                "loss/r1": dis_loss_r1,
                "loss/r2": dis_loss_r2,
                "loss/cp": dis_loss_cp,
                "logits/real": logits_real.detach().mean(),
                "logits/fake": logits_fake.detach().mean(),
            }
        else:
            ot_scale = self.get_ot_scale(step)

            logits_real = self.dis(latents_tgt_real_aug, labels, timesteps_tgt)
            logits_fake = self.dis(self.grad_norm(self.gen_dis_grad_capture(latents_tgt_pred_aug.clone())), labels, timesteps_tgt)

            gen_loss_adv = F.softplus(-(logits_fake - logits_real)).mean()
            gen_loss_ot = (self.gen_ot_grad_capture(latents_tgt_pred.clone()) - latents_src).square().mean([1,2,3]).mul(ot_scale / weighting).mean()

            if self.cls is not None:
                cg_scale = self.config.adversarial.get("cg_scale", 0.0)
                cg_flow = self.config.cls.get("flow", False)
                cg_timestep_lo = self.config.adversarial.get("cg_timestep_lo", 0)
                cg_timestep_hi = self.config.adversarial.get("cg_timestep_hi", 1)

                if cg_flow:
                    latents_cg = self.interpolate(latents_tgt_pred.clone(), noises_cg, timesteps_cg)
                else:
                    latents_cg = latents_tgt_pred.clone()

                logits_cg = self.cls(latents_cg, timesteps_cg)
                gen_loss_cg = F.cross_entropy(logits_cg, labels, reduction="none").mul(cg_scale)
                gen_loss_cg = torch.where((timesteps_cg >= cg_timestep_lo) & (timesteps_cg <= cg_timestep_hi), gen_loss_cg, 0)
                gen_loss_cg = gen_loss_cg.mean()
            else:
                gen_loss_cg = torch.tensor(0.0, device=latents.device)

            loss_dict = {
                "loss/total": (gen_loss_adv + gen_loss_ot + gen_loss_cg) / accu,
                "loss/gen": gen_loss_adv,
                "loss/ot": gen_loss_ot,
                "loss/cg": gen_loss_cg,
                "logits/real": logits_real.detach().mean(),
                "logits/fake": logits_fake.detach().mean(),
            }

        visual_dict = {
            "latents_pred": latents_tgt_pred,
        }

        return loss_dict, visual_dict


    def optimizer_step(self, step):
        stats_optimizer = {}

        if not self.is_update_step(step):
            return stats_optimizer

        if self.dis_lr_schedule is not None:
            apply_lr(self.dis_optimizer, self.dis_lr_schedule, step)
        if self.gen_lr_schedule is not None:
            apply_lr(self.gen_optimizer, self.gen_lr_schedule, step)

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

        self.dis_optimizer.zero_grad()
        self.gen_optimizer.zero_grad()

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
                **self.get_gradient_statistics(),
                "lr": self.optimizer.param_groups[0]["lr"],
                "ot_scale": self.get_ot_scale(step),
                "gp_scale": self.get_gp_scale(step),
                "aug_p": self.get_aug_p(step),
            },
        )

    @torch.no_grad()
    @global_rank_zero_only
    @log_on_entry
    def write_visuals(self, step, batch, inputs, results):
        max_batch_size = self.config.writer.max_batch_size
        latents_pred = results["latents_pred"]
        images = batch["image"]
        images_pred = self.vae_decode(latents_pred)
        self.writer.log_images(
            step=step,
            images={
                "image/data": images[:max_batch_size],
                "image/pred": images_pred[:max_batch_size],
            },
            captions={
                "image/data": list(map(str, batch["label"]))[:max_batch_size],
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
    
    def get_gradient_statistics(self):
        results = {}
        if self.gen_dis_grad_capture.grad is not None:
            results["grad/gen_dis_norm"] = self.gen_dis_grad_capture.grad.norm()
        if self.gen_ot_grad_capture.grad is not None:
            results["grad/gen_ot_norm"] = self.gen_ot_grad_capture.grad.norm()
        return results

    def get_model_statistics(self):
        param_mean = [p.detach().mean().item() for p in self.gen.parameters()]
        param_std = [p.detach().std().item() for p in self.gen.parameters()]
        num_buffers = sum(b.numel() for b in self.gen.buffers())
        num_params = sum(p.numel() for p in self.gen.parameters())
        num_params_trainable = sum(p.numel() for p in self.gen.parameters() if p.requires_grad)
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

    # ------------------------------ Inference ------------------------------ #

    @torch.no_grad()
    def validation(self, step, image_batch, inputs, results):
        noises = inputs["noises_src"]

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
        )
        self.writer.log_images(
            step=step,
            images={
                "image/validation": samples,
            },
            value_range=(-1, 1),
        )

    @torch.no_grad()
    def inference(
        self,
        noises: Tensor,
        labels: Tensor,
    ) -> Tensor:
        steps = self.config.adversarial.get("steps", 1)
        if steps == "any":
            steps = 8

        device = noises.device
        timesteps = torch.linspace(1.0, 0.0, steps+1, device=device)

        model = self.ema if hasattr(self, "ema") else self.gen
        model.eval()

        latents = noises
        for timesteps_src, timesteps_tgt in zip(timesteps[:-1], timesteps[1:]):
            timesteps_src = timesteps_src.repeat(len(noises))
            timesteps_tgt = timesteps_tgt.repeat(len(noises))
            outputs = model(latents, labels, timesteps_src, timesteps_tgt)
            if self.config.gen.get("pred_type") == "v":
                latents = latents - (timesteps_src - timesteps_tgt).view(-1, 1, 1, 1) * outputs
            else:
                latents = outputs

        model.train()
        samples = self.vae_decode(latents)
        return samples

    # ----------------------------- Evaluation ------------------------------ #

    @torch.no_grad()
    def evaluate(self, step):
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
            noises = torch.randn(
                size=[len(labels), 4, resolution // 8, resolution // 8],
                device=device,
            )

            samples = self.inference(
                noises=noises,
                labels=labels,
            )
            samples = samples * 0.5 + 0.5

            self.evaluator.update(preds=samples, target=None)

        results = self.evaluator.compute_and_reset()
        results = {f"eval/{key}": value.item() for key, value in results.items()}

        self.writer.log_metrics(step=step, metrics=results)
        self.logger.info(f"Evaluation Results: {results}")

    # ----------------------------- Persistence ----------------------------- #

    @log_on_entry
    def save_checkpoint(self, step):
        self.persistence.save_model(
            step=step,
            name="gen",
            config=self.config.gen.model,
            states=self.gen.module.state_dict(),
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
        self.persistence.save_model(
            step=step,
            name="dis",
            config=self.config.dis.model,
            states=self.dis.module.state_dict(),
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
        if self.config.adversarial.grad_norm:
            self.persistence.save_optimizer(
                step=step,
                name="grad_norm",
                states=self.grad_norm.state_dict(),
                blocking=True,
            )
