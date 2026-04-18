from .sit_mod import SiTMod


class Generator(SiTMod):
    
    def forward(self, x, y, t):
        # flip t and v,
        # because our training code is x0=image, x1=noise,
        # but sit was trained with x0=noise, x1=image.
        return -super().forward(x, (1.0 - t), y)
