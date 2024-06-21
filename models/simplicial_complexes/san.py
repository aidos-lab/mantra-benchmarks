from torch import nn

from ..model_types import ModelType
from pydantic import BaseModel
from topomodelx.nn.simplicial.san import SAN as SANBack


class SANConfig(BaseModel):
    type: ModelType = ModelType.SAN
    in_channels: int
    hidden_channels: int
    out_channels: int
    n_filters: int = 2
    order_harmonic: int = 5
    epsilon_harmonic: float = 1e-1
    n_layers: int = 2


class SAN(nn.Module):
    """Simplicial Attention Network (SAN) implementation.

        Original paper: Simplicial Attention Neural Networks (https://arxiv.org/pdf/2203.07485)

        Parameters
        ----------
        in_channels : int
            Dimension of input features.
        hidden_channels : int
            Dimension of hidden features.
        out_channels : int
            Dimension of output features.
        n_filters : int, default = 2
            Approximation order for simplicial filters.
        order_harmonic : int, default = 5
            Approximation order for harmonic convolution.
        epsilon_harmonic : float, default = 1e-1
            Epsilon value for harmonic convolution.
        n_layers : int, default = 2
            Number of message passing layers.
        """

    def __init__(
            self,
            config: SANConfig
    ):
        super().__init__()
        self.san_backbone = SANBack(
            config.in_channels,
            config.hidden_channels,
            config.out_channels,
            config.n_filters,
            config.order_harmonic,
            config.epsilon_harmonic,
            config.n_layers,
        )

    def forward(self, batch):
        pass
