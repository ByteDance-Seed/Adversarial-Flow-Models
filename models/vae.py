from typing import NamedTuple
import torch
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.utils.accelerate_utils import apply_forward_hook


class AutoencoderOutput(NamedTuple):
    sample: torch.Tensor
    posterior: DiagonalGaussianDistribution


class EncoderOutput(NamedTuple):
    latent: torch.Tensor
    posterior: DiagonalGaussianDistribution


class DecoderOutput(NamedTuple):
    sample: torch.Tensor


class AutoencoderKLWrapper(AutoencoderKL):
    def load_state_dict(self, state_dict, *args, **kwargs):
        # Newer version of diffusers changed the model keys.
        # This causes incompatibility with old checkpoints.
        # They provided a method for conversion.
        # We call conversion before loading state_dict.
        self._convert_deprecated_attention_blocks(state_dict)
        return super().load_state_dict(state_dict, *args, **kwargs)

    @apply_forward_hook
    def encode(self, x: torch.FloatTensor, sample_posterior: bool = True) -> EncoderOutput:
        """
        Encode a batch of images into latents.

        Args:
            x (`torch.FloatTensor`): Input batch of images.
        """
        if self.use_tiling and (
            x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size
        ):
            return self.tiled_encode(x, return_dict=False).latent_dist

        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self.encoder(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self.encoder(x)

        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        if sample_posterior:
            latent = posterior.sample()
        else:
            latent = posterior.mode()
        return EncoderOutput(latent=latent, posterior=posterior)

    @apply_forward_hook
    def decode(
        self,
        z: torch.FloatTensor,
    ) -> DecoderOutput:
        """
        Decode a batch of images.

        Args:
            z (`torch.FloatTensor`): Input batch of latent vectors.
        """
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample
        return DecoderOutput(sample=decoded)

    def forward(
        self,
        x: torch.FloatTensor,
        sample_posterior: bool = False,
    ) -> AutoencoderOutput:
        r"""
        Args:
            x (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
        """
        latent, posterior = self.encode(x, sample_posterior=sample_posterior)
        sample = self.decode(latent).sample
        return AutoencoderOutput(sample=sample, posterior=posterior)
