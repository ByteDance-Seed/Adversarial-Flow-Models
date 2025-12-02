import torch


class GradientCapture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.grad = None

    def forward(self, x: torch.Tensor):
        def save_grad(grad):
            self.grad = grad
        x.register_hook(save_grad)
        return x
