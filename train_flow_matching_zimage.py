import datetime
import time
import random
from copy import deepcopy
from statistics import mean
from typing import Any, Dict, List
import torch
from diffusers import ZImageTransformer2DModel, AutoencoderKL
from diffusers.models.transformers.transformer_z_image import ZImageTransformerBlock
from einops import rearrange
from omegaconf import OmegaConf
from torch import Tensor
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel,
    FullStateDictConfig,
    FullOptimStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, ToTensor
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
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
from common.partition import partition_by_groups
from common.persistence import PersistenceMixin
from common.seed import set_seed, shift_seed
from common.writers import WriterMixin
from data.t2i_transforms import AreaResize, DivisibleCrop

class FlowMatchingTrainer(Entrypoint, PersistenceMixin, WriterMixin):
    def entrypoint(self):
        init_torch(cudnn_benchmark=False, timeout=datetime.timedelta(seconds=3600))
        self.configure_persistence()
        self.configure_seed()
        self.configure_dataloaders()
        self.configure_models()
        self.configure_optimizers()
        self.configure_accumulators()
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
        self.configure_dit_model(device="cpu")
        self.configure_ema_model(device="cpu")
        self.configure_vae_model()
        self.configure_text_model(device="cpu")
        self.configure_dit_fsdp_model()
        self.configure_text_fsdp_model()

    @log_on_entry
    def configure_dit_model(self, device=get_device()):
        # Create dit model.
        self.dit = ZImageTransformer2DModel.from_pretrained(self.config.dit.path).to(device)
        self.dit.enable_gradient_checkpointing()

        # Load dit checkpoint.
        checkpoint = self.config.dit.get("checkpoint", None)
        if self.resume:
            checkpoint = self.resume.models["dit"].states.path
        if checkpoint:
            state = torch.load(download(checkpoint), map_location=device)
            self.dit.load_state_dict(state, strict=self.config.dit.get("strict", True))

        # Print model size.
        num_params = sum(p.numel() for p in self.dit.parameters() if p.requires_grad)
        self.logger.info(f"dit trainable parameters: {num_params:,}")

    @log_on_entry
    def configure_ema_model(self, device=get_device()):
        # Skip if not needed.
        if not self.config.get("ema"):
            return

        # Create ema model.
        self.ema = deepcopy(self.dit).to(device)

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
        dtype = torch.bfloat16
        self.vae = AutoencoderKL.from_pretrained(self.config.vae.path)
        self.vae.requires_grad_(False).eval()
        self.vae.to(device=get_device(), dtype=dtype)

    @log_on_entry
    def configure_text_model(self, device=get_device()):
        dtype = torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer.path)
        self.text_encoder = AutoModelForCausalLM.from_pretrained(self.config.text_encoder.path)
        self.text_encoder.requires_grad_(False).eval()
        self.text_encoder.to(device=device, dtype=dtype)

    # --------------------------------- DDP -------------------------------- #

    @log_on_entry
    def configure_dit_fsdp_model(self):
        self.dit = self.wrap_dit_with_fsdp(self.dit)
        self.ema = self.wrap_dit_with_fsdp(self.ema)

    def wrap_dit_with_fsdp(self, model):
        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            return recurse or isinstance(module, ZImageTransformerBlock)

        return FullyShardedDataParallel(
            model,
            auto_wrap_policy=custom_auto_wrap_policy,
            device_id=get_local_rank(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            sync_module_states=True,
            forward_prefetch=True,
            limit_all_gathers=False,  # False for ZERO2.
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            ),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        )

    def configure_text_fsdp_model(self):
        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            return recurse or isinstance(module, Qwen3DecoderLayer)

        self.text_encoder = FullyShardedDataParallel(
            self.text_encoder,
            device_id=get_local_rank(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            sync_module_states=False,
            forward_prefetch=True,
            limit_all_gathers=False,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        )

    # ------------------------------ Optimizer ------------------------------ #

    @log_on_entry
    def configure_optimizers(self):
        optimizer_config = OmegaConf.to_container(self.config.optimizer, resolve=True)

        lr = optimizer_config.pop("lr")
        weight_decay = optimizer_config.pop("weight_decay", 0.01)
        betas = optimizer_config.pop("betas", (0.9, 0.95))

        self.optimizer = AdamW(params=self.dit.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, fused=True)

    # ----------------------------- Dataloaders ----------------------------- #

    def configure_dataloaders(self):
        # Create dataset.
        self.image_dataset = create_dataset(
            path=self.config.data.train.dataset.path,
            seed=self.seed,
            image_transform=Compose(
                [
                    AreaResize(
                        max_area=self.config.data.train.dataset.resolution**2,
                        downsample_only=True,
                    ),
                    DivisibleCrop((16, 16)),
                    Normalize(0.5, 0.5),
                ]
            ),
        )

        # Create dataloader
        self.dataloader = DataLoader(
            dataset=self.image_dataset,
            batch_size=self.config.data.train.dataloader.batch_size,
            num_workers=self.config.data.train.dataloader.num_workers,
            prefetch_factor=self.config.data.train.dataloader.prefetch_factor,
            pin_memory=True,
            pin_memory_device=str(get_device()),
            collate_fn=lambda x: x
        )

        # Set up data iterator.
        self.dataloader_iter = iter(self.dataloader)

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

        # Loop over the dataloader.
        while step < last_step:

            # Get batch.
            dataloading_start_time = time.perf_counter()
            image_batch = next(self.dataloader_iter)
            dataloading_time = time.perf_counter() - dataloading_start_time

            # Prepare inputs.
            inputs = self.prepare_input(image_batch)

            # # Train step.
            losses, results = self.training_step(step, **inputs)

            # # Loss backward.
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
                self.validation(step, image_batch, inputs, results, use_ema=False)
                self.validation(step, image_batch, inputs, results, use_ema=True)

            # Update progress bar.
            pbar.set_postfix(
                {
                    "loss": losses["loss/total"].item(),
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "img": len(image_batch),
                    "t_data": dataloading_time,
                }
            )
            pbar.update()

            step += 1

    def prepare_input(
        self, image_batch: Dict[str, Any]
    ) -> Dict[str, Tensor]:
        device = get_device()

        text_dropout = self.config.text_encoder.dropout

        images = [sample["image"].to(device) for sample in image_batch]
        texts = [sample["text"] for sample in image_batch]
        texts = [text if random.random() >= text_dropout else "" for text in texts]

        latents = self.vae_encode(images)
        noises = [torch.randn_like(latent) for latent in latents]

        timestep_type = self.config.training.get("timesteps", "uniform")
        if timestep_type == "uniform":
            timesteps = torch.rand([len(images)], device=device)
        elif timestep_type == "logistic_normal":
            timesteps = torch.randn([len(images)], device=device).sigmoid()

        text_embeds = self.text_encode(texts)

        timesteps = self.shift_timesteps(timesteps)

        return {
            "latents": latents,
            "noises": noises,
            "text_embeds": text_embeds,
            "timesteps": timesteps,
        }

    def shift_timesteps(self, timesteps):
        shift = self.config.training.get("timestep_shift", 1/3)
        return (timesteps * shift) / (1 + (shift - 1) * timesteps)

    def interpolate(self, latents, noises, timesteps):
        # Z-Image uses reverse interpolation, x0=noise, x1=image
        return [
            (1 - t) * x_0 + (t) * x_1
            for x_1, x_0, t in zip(latents, noises, timesteps)
        ]

    def velocity(self, latents, noises, timesteps):
        # Z-Image uses reverse interpolation, x0=noise, x1=image
        return [
            x_1 - x_0
            for x_1, x_0 in zip(latents, noises)
        ]

    def training_step(
        self,
        step: int,
        *,
        latents: List[Tensor],
        noises: List[Tensor],
        text_embeds: List[Tensor],
        timesteps: Tensor,
    ):
        latents_noised = self.interpolate(latents, noises, timesteps)
        v_true = self.velocity(latents, noises, timesteps)
        v_pred = self.dit(latents_noised, timesteps, text_embeds).sample
        loss = [F.mse_loss(v_p.float(), v_t) for v_p, v_t in zip(v_pred, v_true)]
        loss = sum(loss) / len(loss)

        loss_dict = {
            "loss/total": loss,
        }

        visual_dict = {
            "latents_pred": [x_t + (1 - t) * v.detach() for x_t, v, t in zip(latents_noised, v_pred, timesteps)],
        }

        return loss_dict, visual_dict


    def optimizer_step(self, step):
        stats_optimizer = {}
        if self.config.training.gradient_clip is not None:
            stats_optimizer["grad_norm"] = clip_grad_norm_(
                self.dit,
                self.config.training.gradient_clip
            )

        self.optimizer.step()
        self.dit.zero_grad()
        return stats_optimizer

    def ema_step(self, step):
        if hasattr(self, "ema"):
            for tgt, src in zip(self.ema.parameters(), self.dit.parameters()):
                tgt.data.lerp_(src.data.to(tgt), 1 - self.config.ema.decay)

    # -------------------------------- VAE and Text ------------------------------- #

    @torch.no_grad()
    def text_encode(self, texts: List[str]) -> List[Tensor]:
        texts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            for text in texts
        ]
        text_inputs = self.tokenizer(
            texts,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        text_masks = text_inputs.attention_mask.to(self.text_encoder.device).bool()
        text_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_masks,
            output_hidden_states=True,
        ).hidden_states[-2]
        text_embeds = [emb[msk] for emb, msk in zip(text_embeds, text_masks)]
        return text_embeds

    @torch.no_grad()
    def vae_encode(self, samples: List[Tensor]) -> List[Tensor]:
        scale = self.vae.config.scaling_factor
        shift = self.vae.config.shift_factor
        latents = []
        for sample in samples:
            sample = sample.to(self.vae.dtype)
            sample = sample.clamp(-1, 1)
            sample = rearrange(sample, "c h w -> 1 c h w")
            latent = self.vae.encode(sample).latent_dist.sample()
            latent = (latent - shift) * scale
            latent = latent.float()
            latent = rearrange(latent, "f c h w -> c f h w")
            latents.append(latent)
        return latents

    @torch.no_grad()
    def vae_decode(self, latents: List[Tensor]) -> List[Tensor]:
        scale = self.vae.config.scaling_factor
        shift = self.vae.config.shift_factor
        samples = []
        for latent in latents:
            latent = rearrange(latent, "c f h w -> f c h w")
            latent = latent.to(self.vae.dtype)
            latent = latent / scale + shift
            sample = self.vae.decode(latent).sample
            sample = sample.clamp(-1, 1)
            sample = sample.float()
            sample = sample.squeeze(0)
            samples.append(sample)
        return samples

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
                "image/data": [sample["text"] for sample in batch[:max_batch_size]],
                "image/pred": list(map(str, inputs["timesteps"].tolist()))[:max_batch_size],
            },
            value_range=(-1, 1),
        )

    # ------------------------------ Statistics ----------------------------- #

    def get_latents_statistics(self, inputs):
        latents = inputs["latents"]
        return {
            "latents/mean": sum([latent.mean() for latent in latents]) / len(latents),
            "latents/std": sum([latent.std() for latent in latents]) / len(latents),
        }

    def get_model_statistics(self):
        param_mean = [p.detach().mean().item() for p in self.dit.parameters()]
        param_std = [p.detach().std().item() for p in self.dit.parameters()]
        num_buffers = sum(b.numel() for b in self.dit.buffers())
        num_params = sum(p.numel() for p in self.dit.parameters())
        num_params_trainable = sum(p.numel() for p in self.dit.parameters() if p.requires_grad)
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

    def ema_swap(self):
        # Swap ema and dit parameters.
        if hasattr(self, "ema"):
            for ema, dit in zip(self.ema.parameters(), self.dit.parameters()):
                dit_data = dit.data.clone()
                dit.data.copy_(ema.data)
                ema.data.copy_(dit_data)

    @torch.no_grad()
    def validation(self, step, image_batch, inputs, results, use_ema=False):
        B = self.config.validation.max_batch_size

        noises = inputs["noises"][:B]
        text_embeds = inputs["text_embeds"][:B]

        if use_ema:
            self.ema_swap()

        samples = self.inference(
            noises=noises,
            text_embeds=text_embeds,
        )

        if use_ema:
            self.ema_swap()

        postfix = "_ema" if use_ema else ""
        self.writer.log_images(
            step=step,
            images={
                f"image/validation{postfix}": samples,
            },
            value_range=(-1, 1),
        )

    @torch.no_grad()
    def inference(
        self,
        noises: List[Tensor],
        text_embeds: List[Tensor],
    ) -> List[Tensor]:
        steps = self.config.validation.get("steps", 25)
        device = get_device()

        model = self.dit
        model.eval()
        
        timesteps = torch.linspace(0.0, 1.0, steps+1, device=device)
        timesteps = self.shift_timesteps(timesteps)

        latents = noises
        for timesteps_src, timesteps_tgt in zip(timesteps[:-1], timesteps[1:]):
            timesteps_src = timesteps_src.repeat(len(noises))
            timesteps_tgt = timesteps_tgt.repeat(len(noises))
            outputs = model(latents, timesteps_src, text_embeds).sample
            latents = [x_t + (t_tgt - t_src) * v for x_t, v, t_tgt, t_src in zip(latents, outputs, timesteps_tgt, timesteps_src)]

        model.train()
        samples = self.vae_decode(latents)
        return samples

    # ----------------------------- Persistence ----------------------------- #

    @log_on_entry
    def save_checkpoint(self, step):
        FullyShardedDataParallel.set_state_dict_type(
            module=self.dit,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            optim_state_dict_config=FullOptimStateDictConfig(
                offload_to_cpu=True, rank0_only=True
            ),
        )
        FullyShardedDataParallel.set_state_dict_type(
            module=self.ema,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            optim_state_dict_config=FullOptimStateDictConfig(
                offload_to_cpu=True, rank0_only=True
            ),
        )

        self.persistence.save_model(
            step=step,
            name="dit",
            config=self.config.dit,
            states=self.dit.state_dict(),
            blocking=True,
        )
        if hasattr(self, "ema"):
            self.persistence.save_model(
                step=step,
                name="ema",
                config=self.config.dit,
                states=self.ema.state_dict(),
                blocking=True,
            )