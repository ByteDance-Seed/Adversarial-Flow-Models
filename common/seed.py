import random
from typing import Optional
import numpy as np
import torch

from common.distributed import get_global_rank


def set_seed(seed: Optional[int], same_across_ranks: bool = False):
    """Function that sets the seed for pseudo-random number generators."""
    if seed is not None:
        seed += get_global_rank() if not same_across_ranks else 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def shift_seed(seed: Optional[int], shift: int) -> Optional[int]:
    """
    Shift the seed by a given amount. Or return None if seed is None.
    """
    return (seed + shift) if seed is not None else None
