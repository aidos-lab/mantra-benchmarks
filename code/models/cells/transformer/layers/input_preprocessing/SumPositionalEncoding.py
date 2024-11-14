from typing import Optional

import torch
import torch.nn as nn
from jaxtyping import Float

from models.cells.transformer.WeightInitialization import (
    WeightInitialization,
    get_initialization_function,
)
from models.cells.transformer.layers.input_preprocessing.BaseInputPreprocessing import (
    BaseInputPreprocessing,
)


class SumPositionalEncoding(BaseInputPreprocessing):
    def __init__(
        self,
        dim_features: int,
        dim_positional_encoding: int,
        hidden_dim: int,
        initialization: WeightInitialization = WeightInitialization.XAVIER_UNIFORM,
    ):
        super().__init__()
        self.linear_features = nn.Linear(dim_features, hidden_dim)
        self.linear_positional = nn.Linear(dim_positional_encoding, hidden_dim)
        self.hidden_dim = hidden_dim
        self.initialization = initialization
        self.reset_parameters()

    def reset_parameters(self, gain: float = 1.0) -> None:
        init_fn = get_initialization_function(self.initialization, gain)
        init_fn(self.linear_features.weight)
        init_fn(self.linear_positional.weight)

    def forward(
        self,
        x: Float[torch.Tensor, "..."],
        positional_encoding: Optional[Float[torch.Tensor, "..."]],
    ):
        if positional_encoding is None:
            raise ValueError(
                "Positional encoding is required for SumPositionalEncoding"
            )
        return self.linear_features(x) + self.linear_positional(
            positional_encoding
        )
