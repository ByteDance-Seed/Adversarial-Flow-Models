import torch
from torch import nn


from .sit_mod import SiTMod, get_norm


class Discriminator(SiTMod):
    def __init__(self, *, norm_type="rms", **kwargs):
        super().__init__(norm_type=norm_type, **kwargs)
        self.dis_embed = nn.Parameter(torch.randn([kwargs["hidden_size"]]) * 0.02)
        self.final_layer = nn.Sequential(
            get_norm(norm_type)(kwargs["hidden_size"], elementwise_affine=False, eps=kwargs.pop("eps", 1e-6)),
            nn.Linear(kwargs["hidden_size"], 1, bias=False),
        )

    def forward(self, x, y, t):
        # flip t,
        # because our training code is x0=image, x1=noise,
        # but sit was trained with x0=noise, x1=image.
        t = 1.0 - t

        x = self.x_embedder(x) + self.pos_embed.type_as(x)  # (N, T, D), where T = H * W / patch_size ** 2
        c = self.y_embedder(y, self.training).type_as(x)    # (N, D)
        if hasattr(self, "t_embedder"):
            c = c + self.t_embedder(t)
        c = c.type_as(x)

        d = self.dis_embed.view(1, 1, -1)       # (1, 1, d)
        d = d.repeat(x.size(0), 1, 1)           # (N, 1, d)
        d = d.type_as(x)
        x = torch.cat([d, x], dim=1)            # (N, T+1, D)

        for block in self.blocks:
            x = block(x, c)                     # (N, T+1, D)
        x = x[:, :1, :]                         # (N, 1, D)
        x = self.final_layer(x)                 # (N, 1, 1)
        return x.view(-1)                       # (N,)
