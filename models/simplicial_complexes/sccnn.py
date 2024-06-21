from typing import Optional

from torch import nn

from .readouts.SumReadout import SumReadout
from ..model_types import ModelType
from pydantic import BaseModel
from topox.models.sccnn import SCCNNCustom


class SCCNNConfig(BaseModel):
    type: ModelType = ModelType.SCCNN
    in_channels: tuple[int, ...]
    hidden_channels_all: tuple[int, ...] = (64, 64, 64)
    out_channels: int
    conv_order: int = 1
    sc_order: int = 3
    aggr_norm: bool = False
    n_layers: int = 2
    update_func: Optional[str] = None


class SCCNN(nn.Module):
    """SCCNN implementation for complex classification.

       Original paper: Convolutional Learning on Simplicial Complexes (https://arxiv.org/pdf/2301.11163)

        Note: In this task, we can consider the output on any order of simplices for the
        classification task, which of course can be amended by a readout layer.

        Parameters
        ----------
        in_channels : tuple of int
            Dimension of input features on (nodes, edges, faces).
        hidden_channels_all : tuple of int
            Dimension of features of hidden layers on (nodes, edges, faces).
        out_channels : int
            Dimension of output features.
        conv_order : int
            Order of convolutions, we consider the same order for all convolutions.
        sc_order : int
            Order of simplicial complex.
        aggr_norm : bool, optional
            Whether to normalize the aggregation (default: False).
        update_func : str, optional
            Update function for the simplicial complex convolution (default: None).
        n_layers : int, optional
            Number of layers (default: 2).
        """

    def __init__(
            self,
            config: SCCNNConfig
    ):
        super().__init__()
        self.sccnn_backbone = SCCNNCustom(
            config.in_channels,
            config.hidden_channels_all,
            config.conv_order,
            config.sc_order,
            config.aggr_norm,
            config.update_func,
            config.n_layers,
        )
        self.readout_0 = SumReadout(config.hidden_channels_all[0], config.out_channels)
        self.readout_1 = SumReadout(config.hidden_channels_all[1], config.out_channels)
        self.readout_2 = SumReadout(config.hidden_channels_all[2], config.out_channels)

    def forward(self, batch):
        x = batch.x
        connectivity_matrices = batch.connectivity
        x_belonging = batch.x_belonging
        x_0, x_1, x_2 = x[0], x[1], x[2]
        x_bel_0, x_bel_1, x_bel_2 = x_belonging[0], x_belonging[1], x_belonging[2]
        x_all = (x_0, x_1, x_2)
        laplacian_all = (
            connectivity_matrices['hodge_laplacian_0'],
            connectivity_matrices['down_laplacian_1'],
            connectivity_matrices['up_laplacian_1'],
            connectivity_matrices['down_laplacian_2'],
            connectivity_matrices['up_laplacian_2'],
        )
        incidence_all = (connectivity_matrices['incidence_1'], connectivity_matrices['incidence_2'])

        x_0, x_1, x_2 = self.sccnn_backbone(x_all, laplacian_all, incidence_all)
        out_0 = self.readout_0(x_0, x_bel_0)
        out_1 = self.readout_1(x_1, x_bel_1)
        out_2 = self.readout_2(x_2, x_bel_2)
        return out_0 + out_1 + out_2
