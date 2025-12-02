import torch

from .dit import DiT


class GeneratorDeep(DiT):
    def __init__(self, repeat: int, **kwargs):
        super().__init__(**kwargs)
        self.repeat = repeat

    def forward(self, x, y, *args, **kwargs):
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        c = self.y_embedder(y, self.training)    # (N, D)

        for t in torch.arange(1, 0, -1/self.repeat).tolist():
            t = torch.full([len(c)], t, device=c.device, dtype=c.dtype)
            c_i = c + self.t_embedder(t * 1000)  # (N, D), multiply 1000 to align scaling with original DiTs.
            for block in self.blocks:
                x = block(x, c_i)                # (N, T, D)

        x = self.final_layer(x, c)               # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x
