import torch
import torch.nn as nn
import torch.distributed as dist


class GradientNormalization(nn.Module):
    def __init__(self, ema_decay=0.9, eps=1e-8, target_scale=1.0):
        super().__init__()
        self.ema_decay = ema_decay
        self.eps = eps
        self.target_scale = target_scale
        self.register_buffer("square_avg", torch.tensor(0.0))

    def forward(self, x):
        return _GradientNormalizationFn.apply(
            x,
            self.square_avg,
            self.ema_decay,
            self.eps,
            self.target_scale
        )


class _GradientNormalizationFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, square_avg, ema_decay, eps, target_scale):
        ctx.square_avg = square_avg
        ctx.ema_decay = ema_decay
        ctx.eps = eps
        ctx.target_scale = target_scale
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        square_avg = ctx.square_avg
        ema_decay = ctx.ema_decay
        eps = ctx.eps
        target_scale = ctx.target_scale

        # Multiply n here is equivalent to divide by sqrt(n) later in the paper.
        # But this is better because it makes grad_sq_sum independent of local batch size.
        grad_sq_sum = grad_output.square().sum() * grad_output.numel()

        # Here, we compute avg not sum for distributed training.
        # This is only to exchange the local grad_sq_sum.
        # We still want it to be in local scale, not global scale.
        if dist.is_initialized():
            dist.all_reduce(grad_sq_sum, op=dist.ReduceOp.AVG)

        square_avg.lerp_(grad_sq_sum, 1 - ema_decay)
        scale = square_avg.sqrt() + eps
        grad_output = grad_output * (target_scale / scale)
        return grad_output, None, None, None, None


if __name__ == "__main__":
    # Define a loss scale.
    scale = 1000

    # Without grad norm.
    linear = nn.Linear(10, 10)
    x = torch.ones([100, 10])
    x = linear(x)
    x.mean().mul(scale).backward()
    # The gradient is influenced by scale.
    # You can change the scale to see it indeed is influenced by scale.
    print(linear.weight.grad)

    # With grad norm.
    linear = nn.Linear(10, 10)
    x = torch.ones([100, 10])
    x = linear(x)
    x = GradientNormalization(ema_decay=0.0)(x)
    x.mean().mul(scale).backward()
    # The gradient is no longer influenced by scale.
    # The gradient is always equal to the scale=1 case.
    print(linear.weight.grad) 
