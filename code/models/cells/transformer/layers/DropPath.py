import math

import torch
from jaxtyping import Float
from torch import nn


def drop_path(x: Float[torch.Tensor, "..."], p: float = 0.1, training: bool = False) -> Float[torch.Tensor, "..."]:
    """
    DropPath operation introducted in [1].
    Args:
        x (floatTensor): Input tensor.
        p (float, optional): Probability of the path to be dropped. Default:
            ``0.1``.
        training (bool, optional): Whether the module is in training mode. If
            ``False``, this method would return the inputs directly.
    References:
        1. Larsson et al. (https://arxiv.org/abs/1605.07648)
    """
    if math.isclose(p, 0) or not training:
        return x
    prob = 1 - p
    size = (x.size(0),) + (1,) * (x.dim() - 1)
    rand = prob + torch.rand(size, dtype=x.dtype, device=x.device)
    return x / prob * rand.floor()


class DropPath(nn.Module):
    """
    DropPath operation introducted in [1].
    Args:
        p (float, optional): Probability of the path to be dropped. Default:
            ``0.1``.
    References:
        1. Larsson et al. (https://arxiv.org/abs/1605.07648)
    """

    def __init__(self, p=0.1):
        super(DropPath, self).__init__()
        self._p = p

    def __repr__(self) -> str:
        return '{}(p={})'.format(self.__class__.__name__, self._p)

    def forward(self, x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        return drop_path(x, self._p, self.training)
