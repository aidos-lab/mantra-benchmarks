from typing import Optional

import torch
from jaxtyping import Float
from torch import nn

from models.cells.transformer.WeightInitialization import (
    WeightInitialization,
    get_initialization_function,
)
from models.cells.transformer.layers.input_preprocessing.BaseInputPreprocessing import (
    BaseInputPreprocessing,
)


class ConcatenationPositionalEncoding(BaseInputPreprocessing):
    def __init__(
            self,
            dim_features: int,
            dim_positional_encoding: int,
            hidden_dim: int,
            initialization: WeightInitialization = WeightInitialization.XAVIER_UNIFORM,
    ):
        super().__init__()
        self.linear = nn.Linear(dim_features + dim_positional_encoding, hidden_dim)
        self.hidden_dim = hidden_dim
        self.initialization = initialization
        self.reset_parameters()

    def reset_parameters(self, gain: float = 1.0):
        init_fn = get_initialization_function(self.initialization, gain)
        init_fn(self.linear.weight)

    def forward(
            self,
            x: Float[torch.Tensor, "..."],
            positional_encoding: Optional[Float[torch.Tensor, "..."]],
    ):
        return self.linear(torch.cat((x, positional_encoding), dim=-1))
