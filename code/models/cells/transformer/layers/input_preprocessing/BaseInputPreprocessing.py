from abc import ABC, abstractmethod
from typing import Optional

import torch
from jaxtyping import Float
from torch import nn


class BaseInputPreprocessing(nn.Module, ABC):
    @abstractmethod
    def forward(
            self,
            x: Float[torch.Tensor, "..."],
            positional_encoding: Optional[Float[torch.Tensor, "..."]],
    ):
        pass
