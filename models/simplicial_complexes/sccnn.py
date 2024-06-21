from typing import Optional

from torch import nn

from ..model_types import ModelType
from pydantic import BaseModel
from topox.models.sccnn import SCCNNCustom


class SCCNNConfig(BaseModel):
    type: ModelType = ModelType.SCCNN
    in_channels_all: tuple[int]
    hidden_channels_all: tuple[int]
    out_channels: int
    conv_order: int
    sc_order: int
    aggr_norm: bool = False
    update_func: Optional[str] = None,
    n_layers: int = 2,


class SCCNN(nn.Module):
    """SCCNN implementation for complex classification.

       Original paper: Convolutional Learning on Simplicial Complexes (https://arxiv.org/pdf/2301.11163)

        Note: In this task, we can consider the output on any order of simplices for the
        classification task, which of course can be amended by a readout layer.

        Parameters
        ----------
        in_channels_all : tuple of int
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
        super(SCCNN, self).__init__()
        self.sccnn_backbone = SCCNNCustom(
            config.in_channels_all,
            config.hidden_channels_all,
            config.conv_order,
            config.sc_order,
            config.aggr_norm,
            config.update_func,
            config.n_layers,
        )

    def forward(self, batch):
        pass
