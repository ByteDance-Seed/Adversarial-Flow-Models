from torch import nn
from torch.func import jvp, vmap


class DiscriminatorJVP(nn.Module):
    """
    A discriminator wrapper class for handling jvp and vmap.
    This class is used for imagenet training with ddp support.
    Safe to use with ddp(jvpvmap(dis))(x, y, t, dx, dt)
    """

    def __init__(self, dis):
        super().__init__()
        self.dis = dis

    def forward(self, x, y, t, dx, dt):
        
        def dis(x, t):
            return self.dis(x, y, t)

        def dis_jvp(dx, dt):
            return jvp(dis, (x, t), (dx, dt))
    
        def dis_jvp_vmap(dx, dt):
            return vmap(dis_jvp)(dx, dt)

        if x.ndim == dx.ndim:
            o, do = dis_jvp(dx, dt)
        else:
            o, do = dis_jvp_vmap(dx, dt)
        
        return o, do
