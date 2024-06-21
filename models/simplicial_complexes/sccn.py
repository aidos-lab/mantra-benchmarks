from torch import nn
from topomodelx.nn.simplicial.sccn import SCCN as SCCNBack

from ..model_types import ModelType
from pydantic import BaseModel


class SCCNConfig(BaseModel):
    type: ModelType = ModelType.SCCN
    channels: int
    out_channels: int
    max_rank: int
    n_layers: int = 2
    update_func: str = "sigmoid"


class SCCN(nn.Module):
    """Simplicial Complex Convolutional Network Implementation for binary node classification.

        Original paper: Efficient Representation Learning for Higher-Order Data with Simplicial Complexes (https://openreview.net/pdf?id=nGqJY4DODN)

        Parameters
        ----------
        channels : int
            Dimension of features.
        out_channels : int
            Dimension of output features.
        max_rank : int
            Maximum rank of the cells in the simplicial complex.
        n_layers : int
            Number of message passing layers.
        update_func : str
            Activation function used in aggregation layers.
        """

    def __init__(self, config: SCCNConfig):
        super().__init__()
        self.sccn_backbone = SCCNBack(
            channels=config.channels,
            max_rank=config.max_rank,
            n_layers=config.n_layers,
            update_func=config.update_func,
        )

    def forward(self, batch):
        pass
