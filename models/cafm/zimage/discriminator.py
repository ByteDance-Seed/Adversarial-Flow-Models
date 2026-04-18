from typing import Optional
import torch
from torch import nn
from diffusers.models.transformers.transformer_z_image import (
    ZImageTransformer2DModel,
    ZImageTransformerBlock,
)
from torch.func import jvp, vmap

from .logit_proj import LogitProject


def tangent_vmap_stack(tangent_fake, tangent_real):
    if tangent_real is not None:
        return torch.stack([tangent_fake, tangent_real])
    else:
        return tangent_fake

def tangent_vmap_unbind(primal, tangents):
    if tangents.ndim == primal.ndim:
        return tangents, None
    else:
        return tangents.unbind(0)


class ZImageTransformerBlockJVP(ZImageTransformerBlock):
    def forward(
        self,
        x: torch.Tensor,
        x_tangent: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: torch.Tensor,
        adaln_input_tangent: torch.Tensor,
        noise_mask: Optional[torch.Tensor] = None,
        adaln_noisy: Optional[torch.Tensor] = None,
        adaln_clean: Optional[torch.Tensor] = None,
    ):
        forward = super().forward
        def forward_jvp(x_tangent, adaln_input_tangent):
            return jvp(lambda x, adaln_input: forward(x, attn_mask, freqs_cis, adaln_input, noise_mask, adaln_noisy, adaln_clean), (x, adaln_input), (x_tangent, adaln_input_tangent))
        if x_tangent.ndim == x.ndim:
            x, x_tangent = forward_jvp(x_tangent, adaln_input_tangent)
        else:
            x, x_tangent = vmap(forward_jvp)(x_tangent, adaln_input_tangent)
            x = x[0]
        return x, x_tangent


class ZImageTransformer2DModelDiscriminatorJVP(ZImageTransformer2DModel):
    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        siglip_feat_dim=None,  # Optional: set to enable SigLIP support for Omni
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[32, 48, 48],
        axes_lens=[1024, 512, 512],
    ):
        super().__init__(
            all_patch_size=all_patch_size,
            all_f_patch_size=all_f_patch_size,
            in_channels=in_channels,
            dim=dim,
            n_layers=n_layers,
            n_refiner_layers=n_refiner_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            cap_feat_dim=cap_feat_dim,
            siglip_feat_dim=siglip_feat_dim,
            rope_theta=rope_theta,
            t_scale=t_scale,
            axes_dims=axes_dims,
            axes_lens=axes_lens,
        )
        self.noise_refiner = nn.ModuleList(
            [
                ZImageTransformerBlockJVP(
                    1000 + layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.layers = nn.ModuleList(
            [
                ZImageTransformerBlockJVP(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm)
                for layer_id in range(n_layers)
            ]
        )
        self.out = LogitProject(dim=dim, heads=n_heads, head_dim=dim // n_heads)

        del self.all_final_layer

    def forward(
        self,
        x,
        t,        
        cap_feats_input,
        x_tangent_fake,
        t_tangent_fake,
        x_tangent_real = None,
        t_tangent_real = None,
        patch_size: int = 2,
        f_patch_size: int = 1,
    ):
        device = x[0].device
        omni_mode = False
        t_noisy = t_clean = None
        x_pos_offsets = x_noise_mask = x_noise_tensor= cap_noise_mask = siglip_noise_mask = None

        def _input_process(x, t):
            # Single embedding for all tokens
            adaln_input = self.t_embedder(t * self.t_scale).type_as(x[0])

            # Patchify
            (
                x,
                cap_feats,
                x_size,
                x_pos_ids,
                cap_pos_ids,
                x_pad_mask,
                cap_pad_mask,
            ) = self.patchify_and_embed(x, cap_feats_input, patch_size, f_patch_size)

            # X embed & refine
            x_seqlens = [len(xi) for xi in x]
            x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](torch.cat(x, dim=0))  # embed
            x, x_freqs, x_mask, _, _ = self._prepare_sequence(
                list(x.split(x_seqlens, dim=0)), x_pos_ids, x_pad_mask, self.x_pad_token, x_noise_mask, device
            )

            # Cap embed & refine
            cap_seqlens = [len(ci) for ci in cap_feats]
            cap_feats = self.cap_embedder(torch.cat(cap_feats, dim=0))  # embed
            cap_feats, cap_freqs, cap_mask, _, _ = self._prepare_sequence(
                list(cap_feats.split(cap_seqlens, dim=0)), cap_pos_ids, cap_pad_mask, self.cap_pad_token, None, device
            )

            # jvp only support tensor output, so convert them to tensor first.
            x_size = torch.tensor(x_size)
            x_seqlens = torch.tensor(x_seqlens)
            cap_seqlens = torch.tensor(cap_seqlens)

            return (x, adaln_input), (x_mask, x_freqs, x_size, x_seqlens, cap_feats, cap_freqs, cap_mask, cap_seqlens)

        if x_tangent_real is not None:
            _, (x_tangent_real, adaln_input_tangent_real), _ = jvp(_input_process, (x, t), (x_tangent_real, t_tangent_real), has_aux=True)
        else:
            x_tangent_real, adaln_input_tangent_real = None, None

        (x, adaln_input), (x_tangent_fake, adaln_input_tangent_fake), (x_mask, x_freqs, x_size, x_seqlens, cap_feats, cap_freqs, cap_mask, cap_seqlens) = jvp(_input_process, (x, t), (x_tangent_fake, t_tangent_fake), has_aux=True)        

        x_size = x_size.tolist()
        x_seqlens = x_seqlens.tolist()
        cap_seqlens = cap_seqlens.tolist()

        x_tangent = tangent_vmap_stack(x_tangent_fake, x_tangent_real)
        adaln_input_tangent = tangent_vmap_stack(adaln_input_tangent_fake, adaln_input_tangent_real)

        for layer in self.noise_refiner:
            x, x_tangent = (
                self._gradient_checkpointing_func(
                    layer, x, x_tangent, x_mask, x_freqs, adaln_input, adaln_input_tangent, x_noise_tensor, t_noisy, t_clean
                )
                if torch.is_grad_enabled() and self.gradient_checkpointing
                else layer(x, x_tangent, x_mask, x_freqs, adaln_input, adaln_input_tangent, x_noise_tensor, t_noisy, t_clean)
            )

        for layer in self.context_refiner:
            cap_feats = (
                self._gradient_checkpointing_func(layer, cap_feats, cap_mask, cap_freqs)
                if torch.is_grad_enabled() and self.gradient_checkpointing
                else layer(cap_feats, cap_mask, cap_freqs)
            )

        # Siglip embed & refine
        siglip_seqlens = siglip_freqs = siglip_feats = None
        unified_noise_tensor = None

        x_tangent_fake, x_tangent_real = tangent_vmap_unbind(x, x_tangent)

        # Unified sequence
        def _build_unified_sequence(x):
            unified, unified_freqs, unified_mask, _ = self._build_unified_sequence(
                x,
                x_freqs,
                x_seqlens,
                x_noise_mask,
                cap_feats,
                cap_freqs,
                cap_seqlens,
                cap_noise_mask,
                siglip_feats,
                siglip_freqs,
                siglip_seqlens,
                siglip_noise_mask,
                omni_mode,
                device,
            )
            return unified, (unified_freqs, unified_mask)

        if x_tangent_real is not None:
            _, unified_tangent_real, _ = jvp(_build_unified_sequence, (x,), (x_tangent_real,), has_aux=True)
        else:
            unified_tangent_real = None
        
        unified, unified_tangent_fake, (unified_freqs, unified_mask) = jvp(_build_unified_sequence, (x,), (x_tangent_fake,), has_aux=True)
        
        unified_tangent = tangent_vmap_stack(unified_tangent_fake, unified_tangent_real)

        # Main transformer layers
        for layer_idx, layer in enumerate(self.layers):
            unified, unified_tangent = (
                self._gradient_checkpointing_func(
                    layer, unified, unified_tangent, unified_mask, unified_freqs, adaln_input, adaln_input_tangent, unified_noise_tensor, t_noisy, t_clean
                )
                if torch.is_grad_enabled() and self.gradient_checkpointing
                else layer(unified, unified_tangent, unified_mask, unified_freqs, adaln_input, adaln_input_tangent, unified_noise_tensor, t_noisy, t_clean)
            )

        unified_tangent_fake, unified_tangent_real = tangent_vmap_unbind(unified, unified_tangent)

        out, out_jvp_fake = jvp(lambda x: self.out(x, x_seqlens), (unified,), (unified_tangent_fake,))
        out_jvp_real = None
        if unified_tangent_real is not None:
            _, out_jvp_real = jvp(lambda x: self.out(x, x_seqlens), (unified,), (unified_tangent_real,))
        
        return out, out_jvp_fake, out_jvp_real
