from torch import nn

from .sit import SiTBlock, FinalLayer, SiT


def get_norm(norm_type):
    if norm_type == "ln":
        return nn.LayerNorm
    if norm_type == "rms":
        return nn.RMSNorm
    raise NotImplementedError(f"norm_type {norm_type} not implemented")


class SiTBlockMod(SiTBlock):
    def __init__(self, hidden_size, num_heads, norm_type, mlp_ratio=4.0, **block_kwargs):
        super().__init__(hidden_size, num_heads, mlp_ratio, **block_kwargs)
        self.norm1 = get_norm(norm_type)(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = get_norm(norm_type)(hidden_size, elementwise_affine=False, eps=1e-6)


class FinalLayerMod(FinalLayer):
    def __init__(self, hidden_size, patch_size, out_channels, norm_type):
        super().__init__(hidden_size, patch_size, out_channels)
        self.norm_final = get_norm(norm_type)(hidden_size, elementwise_affine=False, eps=1e-6)


class SiTMod(SiT):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        norm_type="ln"
    ):
        super().__init__(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            class_dropout_prob=class_dropout_prob,
            num_classes=num_classes,
            learn_sigma=learn_sigma,
        )
        self.blocks = nn.ModuleList([
            SiTBlockMod(hidden_size, num_heads, mlp_ratio=mlp_ratio, norm_type=norm_type) for _ in range(depth)
        ])
        self.final_layer = FinalLayerMod(hidden_size, patch_size, self.out_channels, norm_type)
        self.initialize_weights()
