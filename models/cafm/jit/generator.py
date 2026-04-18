from .jit import JiT


class Generator(JiT):
    def forward(self, x, y, t):
        # flip t,
        # because our training code is x0=image, x1=noise,
        # but jit was trained with x0=noise, x1=image.
        x_pred = super().forward(x=x, y=y, t=(1 - t))
        # v is computed as x0=image, x1=noise.
        v = (x - x_pred) / t.view(-1, 1, 1, 1).clamp_min(0.05)
        return v
