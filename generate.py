"""
This generate.py is only for AFMs.
CAFMs should use the official sampling code from SiT/JiT.
"""


import datetime
import os
import torch
from torch import Tensor
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from common.config import create_object
from common.decorators import barrier_on_entry, log_on_entry
from common.distributed import (
    get_device,
    get_global_rank,
    get_local_rank,
    get_world_size,
    init_torch,
)
from common.entrypoint import Entrypoint
from common.fs import download, mkdir
from common.partition import partition_by_groups
from common.seed import set_seed


class AdversarialFlowGenerator(Entrypoint):
    def entrypoint(self):
        init_torch(cudnn_benchmark=False, timeout=datetime.timedelta(seconds=3600))
        self.configure_seed()
        self.configure_models()
        self.configure_seed()
        self.inference_loop()

    # ----------------------------- Determinism ----------------------------- #

    def configure_seed(self):
        set_seed(self.config.generation.seed)

    # -------------------------------- Model -------------------------------- #

    def configure_models(self):
        self.configure_gen_model()
        self.configure_vae_model()

    @log_on_entry
    def configure_gen_model(self, device=get_device()):
        # Create gen model.
        self.gen = create_object(self.config.gen.model).to(device)

        # Load gen checkpoint.
        checkpoint = self.config.gen.get("checkpoint", None)
        if checkpoint:
            state = torch.load(download(checkpoint), map_location=device)
            self.gen.load_state_dict(state, strict=self.config.gen.get("strict", True))

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

    # ------------------------------ Inference ------------------------------- #

    @barrier_on_entry
    @torch.no_grad()
    def inference_loop(self):
        output = self.config.generation.output
        mkdir(output)

        labels = torch.arange(0, 1000).repeat_interleave(50).tolist()
        labels = list(enumerate(labels))
        labels = partition_by_groups(labels, get_world_size())[get_global_rank()]

        device = get_device()

        for (i, label) in tqdm(labels, position=get_local_rank()):
            noise = torch.randn([1, 4, 32, 32], device=device, generator=torch.Generator("cuda").manual_seed(i + self.config.generation.seed))
            label = torch.tensor([label], device=device, dtype=torch.long)
            sample = self.inference(noise, label)
            to_pil_image(sample.mul(0.5).add(0.5).clamp(0, 1)[0]).save(os.path.join(output, f"{i:05}.png"))

    @torch.no_grad()
    def vae_decode(self, latents: Tensor) -> Tensor:
        dtype = getattr(torch, self.config.vae.dtype)
        scale = self.config.vae.scaling_factor
        latents = latents.to(dtype)
        latents = latents / scale
        samples = self.vae.decode(latents).sample
        return samples.float()

    @torch.no_grad()
    def inference(
        self,
        noises: Tensor,
        labels: Tensor,
    ) -> Tensor:
        steps = self.config.generation.steps

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
