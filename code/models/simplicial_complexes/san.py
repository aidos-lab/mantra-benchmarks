import torch
from pydantic import BaseModel
from topomodelx.nn.simplicial.san import SAN as SANBack
from torch import nn

from .readouts.SumReadout import SumReadout
from ..model_types import ModelType


class SANConfig(BaseModel):
    type: ModelType = ModelType.SAN
    in_channels: tuple[int, ...] = (1, 2, 1)
    in_channels_backbone: int = 64
    hidden_channels: int = 64
    out_channels: int = 3
    n_filters: int = 2
    order_harmonic: int = 5
    epsilon_harmonic: float = 1e-1
    n_layers: int = 1


class SAN(nn.Module):
    """Simplicial Attention Network (SAN) implementation.

    Original paper: Simplicial Attention Neural Networks (https://arxiv.org/pdf/2203.07485)

    Parameters
    ----------
    in_channels : tuple[int]
        Dimension of input features.
    in_channels_backbone: int
        Dimension of input features for the input of the backbone.
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

    def __init__(self, config: SANConfig):
        super().__init__()
        self.san_backbone = SANBack(
            config.in_channels_backbone,
            config.hidden_channels,
            config.out_channels,
            config.n_filters,
            config.order_harmonic,
            config.epsilon_harmonic,
            config.n_layers,
        )
        self.input_0_proj_needed = (
                config.in_channels[0] != config.in_channels_backbone
        )
        self.input_1_proj_needed = (
                config.in_channels[1] != config.in_channels_backbone
        )
        self.input_2_proj_needed = (
                config.in_channels[2] != config.in_channels_backbone
        )
        if self.input_0_proj_needed:
            self.input_projection_0 = nn.Linear(
                config.in_channels[0], config.in_channels_backbone
            )
        if self.input_1_proj_needed:
            self.input_projection_1 = nn.Linear(
                config.in_channels[1], config.in_channels_backbone
            )
        if self.input_2_proj_needed:
            self.input_projection_2 = nn.Linear(
                config.in_channels[2], config.in_channels_backbone
            )
        self.readout = SumReadout(config.out_channels, config.out_channels)

    def forward(self, batch):
        x = batch.x
        connectivity_matrices = batch.connectivity
        x_belonging = batch.x_belonging
        x_0, x_1, x_2 = x[0], x[1], x[2]
        if self.input_0_proj_needed:
            x_0 = self.input_projection_0(x_0)
        if self.input_1_proj_needed:
            x_1 = self.input_projection_1(x_1)
        if self.input_2_proj_needed:
            x_2 = self.input_projection_2(x_2)
        x_bel_0, x_bel_1, x_bel_2 = (
            x_belonging[0],
            x_belonging[1],
            x_belonging[2],
        )
        x_1 = self.san_backbone(
            x_1,
            connectivity_matrices["up_laplacian_1"],
            connectivity_matrices["down_laplacian_1"],
        )
        # TODO: Not used, we can remove it, but it is written in TopoBenchmarkX
        x_0 = torch.sparse.mm(connectivity_matrices["incidence_1"], x_1)
        out = self.readout(x_1, x_bel_1)
        return out
