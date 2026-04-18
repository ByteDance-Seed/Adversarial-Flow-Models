import torch
from torch import nn

from .dit import DiT


class Classifier(DiT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cls_embed = nn.Parameter(torch.randn([kwargs["hidden_size"]]) * 0.02)
        self.final_layer = nn.Sequential(
            nn.LayerNorm(kwargs["hidden_size"], elementwise_affine=False, eps=1e-6),
            nn.Linear(kwargs["hidden_size"], kwargs.get("num_classes", 1000), bias=True),
        )
        del self.y_embedder

    def forward(self, x, t=None):
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        e = self.t_embedder(t * 1000)            # (N, D), multiply 1000 to align scaling with original DiTs.

        o = self.cls_embed.view(1, 1, -1)        # (1, 1, D)
        o = o.repeat(x.size(0), 1, 1)            # (N, 1, D)
        x = torch.cat([o, x], dim=1)             # (N, T+1, D)

        for block in self.blocks:
            x = block(x, e)                      # (N, T+1, D)
        o = x[:, :1, :]                          # (N, 1, D)
        o = self.final_layer(o)                  # (N, 1, C)
        o = o.squeeze(1)                         # (N, C)
        return o
