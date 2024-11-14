from abc import ABC, abstractmethod

import torch
from jaxtyping import Float
from torch import nn


class BaseReadout(nn.Module, ABC):
    """
    BaseReadout class implementing a layer to perform signals readout. Concretely, it converts the signals on top
    of the different cells into a common signal for the whole skeleton or for the whole cellular_data complex, depending
    on the implementation.
    """

    @abstractmethod
    def forward(
        self,
        x: dict[int, Float[torch.Tensor, "..."], ...],
        x_belongings: dict[int, list[int]],
    ) -> dict[int, Float[torch.Tensor, "..."], ...] | Float[
        torch.Tensor, "..."
    ]:
        """
        :param x: Dict of signals for the different dimensions. The keys are the dimensions.
        :param x_belongings: Dict of signal belongings
        :return: Either a dict of tensors for is_global = False or a tensor otherwise.
        """
        pass

    @property
    @abstractmethod
    def is_global(self):
        pass
