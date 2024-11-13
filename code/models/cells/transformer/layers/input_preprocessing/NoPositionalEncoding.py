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


class NoPositionalEncoding(BaseInputPreprocessing):
    def __init__(
            self,
            dim_features: int,
            hidden_dim: int,
            initialization: WeightInitialization = WeightInitialization.XAVIER_UNIFORM,
    ):
        super().__init__()
        self.linear_features = nn.Linear(dim_features, hidden_dim)
        self.initialization = initialization
        self.reset_parameters()

    def reset_parameters(self, gain: float = 1.0):
        init_fn = get_initialization_function(self.initialization, gain)
        init_fn(self.linear_features.weight)

    def forward(
            self,
            x: Float[torch.Tensor, "..."],
            positional_encoding: Float[torch.Tensor, "..."],
    ):
        return self.linear_features(x)
