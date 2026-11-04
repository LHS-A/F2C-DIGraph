"""Warm-up plus cosine scheduling."""

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int,
    total_epochs: int,
) -> LambdaLR:
    """Return a per-epoch LR scheduler with linear warm-up and cosine decay."""
    warmup_epochs = max(1, int(warmup_epochs))
    total_epochs = max(warmup_epochs + 1, int(total_epochs))

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def distill_weight(epoch: int, total_epochs: int, max_weight: float, warmup_fraction: float = 0.3) -> float:
    """Linear distillation-weight warm-up used by Eq. 16 in the paper."""
    warmup_epochs = max(1, int(math.ceil(warmup_fraction * total_epochs)))
    if warmup_epochs == 1:
        progress = 0.0 if epoch <= 0 else 1.0
    else:
        progress = min(max(float(epoch) / float(warmup_epochs - 1), 0.0), 1.0)
    return progress * float(max_weight)

