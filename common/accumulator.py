"""
Accumulator for logging.
"""

from functools import partial
from typing import Dict, List, Literal, Union
import torch
from torch import Tensor
from torch.distributed import ReduceOp, all_reduce, is_initialized

from common.distributed import get_device


class Accumulator:
    """
    Accumulate values over multiple iterations.
    This is intended for loss logging purposes.
    """

    def __init__(self, mode: Literal["avg", "sum", "min", "max"]):
        self.reset()
        assert mode in ["avg", "sum", "min", "max"]
        self.mode = mode

    def reset(self):
        """
        Reset the accumulator.
        """
        self.val = {}
        self.num = {}

    @torch.no_grad()
    def add(self, **kwargs: Dict[str, Union[int, float, Tensor, List]]):
        """
        Add value to the accumulator.
        """
        for k, v in kwargs.items():
            sv = sum(v) if isinstance(v, list) else v
            nv = len(v) if isinstance(v, list) else 1
            if self.mode == "min":
                self.val[k] = min(self.val.get(k, float("inf")), sv)
            if self.mode == "max":
                self.val[k] = max(self.val.get(k, float("-inf")), sv)
            elif self.mode in ["sum", "avg"]:
                self.val[k] = self.val.get(k, 0) + sv
            self.num[k] = self.num.get(k, 0) + nv

    def get(self) -> Dict[str, Union[float, Tensor]]:
        """
        Get accumulated values.
        """
        if self.mode == "avg":
            return {k: self.val[k] / self.num[k] for k in self.val.keys()}
        elif self.mode in ["sum", "min", "max"]:
            return self.val
        else:
            raise NotImplementedError

    def get_and_reset(self) -> Dict[str, Union[float, Tensor]]:
        """
        Get accumulated value and reset.
        """
        val = self.get()
        self.reset()
        return val


class DistributedAccumulator(Accumulator):
    """
    Accumulate values over multiple iterations and over all GPUs.
    This is intended for loss logging purposes.
    The distributed accumulator must be instantiated on all GPU ranks.
    The method "get" and "get_and_reset" must be invoked on all GPU ranks.
    """

    def get(self) -> Dict[str, Union[float, torch.Tensor]]:
        """
        Get accumulated values from all ranks.
        """
        # Instead of all_reduce on values individually,
        # we merge values to a big tensor and only reduce once.
        if len(self.val.keys()) == 0:
            return {}

        result = {k: [self.val[k], self.num[k]] for k in self.val.keys()}.items()
        device = get_device()
        tensor = torch.cat([torch.as_tensor(v, device=device).view(-1) for _, v in result])
        # global operation when distributed.
        if is_initialized():
            if self.mode in ["sum", "avg"]:
                all_reduce(tensor=tensor, op=ReduceOp.SUM)  # sum plus sum, num plus num
            elif self.mode == "min":
                all_reduce(tensor=tensor, op=ReduceOp.MIN)  # min of min
            elif self.mode == "max":
                all_reduce(tensor=tensor, op=ReduceOp.MAX)  # max of max
            else:
                raise NotImplementedError("Other mode is not supported.")
        # get the final value per key
        if self.mode in ["sum", "min", "max"]:
            tensor = tensor[torch.arange(0, len(tensor), 2)]
        elif self.mode == "avg":
            tensor = (
                tensor[torch.arange(0, len(tensor), 2)] / tensor[torch.arange(1, len(tensor), 2)]
            )
        else:
            raise NotImplementedError("Other mode is not supported.")
        # Split back to original shapes.
        tensor = tensor.split([v.numel() if torch.is_tensor(v) else 1 for _, v in result])
        result = {
            k: t.reshape_as(v) if torch.is_tensor(v) else t.item()
            for t, (k, v) in zip(tensor, result)
        }
        # remove inf values
        for key in result.keys():
            if result[key] == float("inf"):
                del result[key]
        return result
