from torch import nn

from .dit import DiT, TimestepEmbedder


class Generator(DiT):
    def __init__(self, *, use_t_src=False, use_t_tgt=False, **kwargs):
        super().__init__(**kwargs)
        if not use_t_src:
            del self.t_embedder
        if use_t_tgt:
            self.t_tgt_embedder = TimestepEmbedder(kwargs["hidden_size"])
            nn.init.normal_(self.t_tgt_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_tgt_embedder.mlp[2].weight, std=0.02)

    def forward(self, x, y, t_src=None, t_tgt=None):
        x = self.x_embedder(x) + self.pos_embed         # (N, T, D), where T = H * W / patch_size ** 2
        c = self.y_embedder(y, self.training)           # (N, D)
        if hasattr(self, "t_embedder"):
            c = c + self.t_embedder(t_src * 1000)       # (N, D), multiply 1000 to align scaling with original DiTs.
        if hasattr(self, "t_tgt_embedder"):
            c = c + self.t_tgt_embedder(t_tgt * 1000)   # (N, D), multiply 1000 to align scaling with original DiTs.

        for block in self.blocks:
            x = block(x, c)                             # (N, T, D)
        x = self.final_layer(x, c)                      # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                          # (N, out_channels, H, W)
        return x
