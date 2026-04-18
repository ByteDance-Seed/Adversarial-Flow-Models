import torch
from torch import nn

from .dit import DiT


class Discriminator(DiT):
    def __init__(self, *, use_t=False, **kwargs):
        super().__init__(**kwargs)
        self.dis_embed = nn.Parameter(torch.randn([kwargs["hidden_size"]]) * 0.02)
        self.final_layer = nn.Sequential(
            nn.LayerNorm(kwargs["hidden_size"], elementwise_affine=False, eps=kwargs.pop("eps", 1e-6)),
            nn.Linear(kwargs["hidden_size"], 1, bias=False),
        )
        if not use_t:
            del self.t_embedder

    def forward(self, x, y, t=None):
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        c = self.y_embedder(y, self.training)    # (N, D)
        if hasattr(self, "t_embedder"):
            c = c + self.t_embedder(t * 1000)    # (N, D),multiply 1000 to align scaling with original DiTs.

        d = self.dis_embed.view(1, 1, -1)        # (1, 1, D)
        d = d.repeat(x.size(0), 1, 1)            # (N, 1, D)
        x = torch.cat([d, x], dim=1)             # (N, T+1, D)

        for block in self.blocks:
            x = block(x, c)                      # (N, T+1, D)
        x = x[:, :1, :]                          # (N, 1, D)
        x = self.final_layer(x)                  # (N, 1, 1)
        return x.view(-1)                        # (N,)
