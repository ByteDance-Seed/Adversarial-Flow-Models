from .jit import JiT
import torch
from torch import nn


class Discriminator(JiT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.final_layer
        self.proj_out = nn.Sequential(
            nn.RMSNorm(kwargs["hidden_size"], elementwise_affine=False),
            nn.Linear(kwargs["hidden_size"], 1),
        )

    def forward(self, x, y, t):
        """
        x: (N, C, H, W)
        t: (N,)
        y: (N,)
        """
        # flip t,
        # because our training code is x0=image, x1=noise,
        # but jit was trained with x0=noise, x1=image.
        t = 1 - t

        # class and time embeddings
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb

        # forward JiT
        x = self.x_embedder(x)
        x += self.pos_embed

        for i, block in enumerate(self.blocks):
            # in-context
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens += self.in_context_posemb
                x = torch.cat([in_context_tokens, x], dim=1)
            x = block(x, c, self.feat_rope if i < self.in_context_start else self.feat_rope_incontext)

        x = x[:, 0]
        x = self.proj_out(x)
        x = x.reshape(-1)
        return x
