import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from diffusers.models.normalization import RMSNorm


class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int):
        super().__init__()
        self.heads = heads
        self.proj_q = nn.Linear(dim, heads * head_dim, bias=False)
        self.proj_k = nn.Linear(dim, heads * head_dim, bias=False)
        self.proj_v = nn.Linear(dim, heads * head_dim, bias=False)
        self.proj_o = nn.Linear(heads * head_dim, dim, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=1e-5)
        self.k_norm = RMSNorm(head_dim, eps=1e-5)
    
    def forward(self, q, kv, x_seqlens):
        b, l, _ = q.shape
        _, s, _ = kv.shape

        q = rearrange(self.proj_q(q), "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(self.proj_k(kv), "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(self.proj_v(kv), "b n (h d) -> b h n d", h=self.heads)

        q = self.q_norm(q)
        k = self.k_norm(k)

        masks = torch.zeros([b, 1, l, s], dtype=torch.bool, device=q.device)
        for i, x_seqlen in enumerate(x_seqlens):
            masks[i, :, :, :x_seqlen] = True

        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_mem_efficient=False,
            enable_cudnn=False,
            enable_math=True,
        ):
            o = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=masks)
        o = rearrange(o, "b h n d -> b n (h d)")
        o = self.proj_o(o)
        return o


class LogitProject(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int):
        super().__init__()
        self.emb = nn.Parameter(torch.randn([dim]) * 0.02)
        self.norm = RMSNorm(dim, elementwise_affine=False, eps=1e-5)
        self.attn = CrossAttention(dim, heads, head_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )
        self.proj = nn.Linear(dim, 1)

    def forward(self, unified, x_seqlens):
        emb = self.emb.view(1, 1, -1).expand(unified.shape[0], 1, -1).type_as(unified)
        emb = emb + self.attn(self.norm(emb), self.norm(unified), x_seqlens)
        emb = emb + self.mlp(self.norm(emb))
        out = self.proj(self.norm(emb)).view(-1)
        return out
