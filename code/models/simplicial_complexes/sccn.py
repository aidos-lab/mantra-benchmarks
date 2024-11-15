from pydantic import BaseModel
from topomodelx.nn.simplicial.sccn import SCCN as SCCNBack
from torch import nn

from .readouts.SumReadout import SumReadout
from ..model_types import ModelType


class SCCNConfig(BaseModel):
    type: ModelType = ModelType.SCCN
    in_channels: tuple[int, ...] = (1, 2, 1)
    channels: int = 64
    out_channels: int = 3
    max_rank: int = 2
    n_layers: int = 2
    update_func: str = "sigmoid"


class SCCN(nn.Module):
    """Simplicial Complex Convolutional Network Implementation for binary node classification.

    Original paper: Efficient Representation Learning for Higher-Order Data with Simplicial Complexes (https://openreview.net/pdf?id=nGqJY4DODN)

    Parameters
    ----------
    in_channels: tuple[int]
        Dimension of input features.
    channels : int
        Dimension of features backbone.
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
        self.input_0_proj_needed = config.in_channels[0] != config.channels
        self.input_1_proj_needed = config.in_channels[1] != config.channels
        self.input_2_proj_needed = config.in_channels[2] != config.channels
        if self.input_0_proj_needed:
            self.input_projection_0 = nn.Linear(
                config.in_channels[0], config.channels
            )
        if self.input_1_proj_needed:
            self.input_projection_1 = nn.Linear(
                config.in_channels[1], config.channels
            )
        if self.input_2_proj_needed:
            self.input_projection_2 = nn.Linear(
                config.in_channels[2], config.channels
            )
        self.readout_0 = SumReadout(config.channels, config.out_channels)
        self.readout_1 = SumReadout(config.channels, config.out_channels)
        self.readout_2 = SumReadout(config.channels, config.out_channels)

    def forward(self, batch):
        x = batch.x
        connectivity_matrices = batch.connectivity
        x_belonging = batch.x_belonging
        x_bel_0, x_bel_1, x_bel_2 = (
            x_belonging[0],
            x_belonging[1],
            x_belonging[2],
        )
        if self.input_0_proj_needed:
            x[0] = self.input_projection_0(x[0])
        if self.input_1_proj_needed:
            x[1] = self.input_projection_1(x[1])
        if self.input_2_proj_needed:
            x[2] = self.input_projection_2(x[2])
        features = {
            f"rank_{r}": x[r]
            for r in range(self.sccn_backbone.layers[0].max_rank + 1)
        }
        incidences = {
            f"rank_{r}": connectivity_matrices[f"incidence_{r}"]
            for r in range(1, self.sccn_backbone.layers[0].max_rank + 1)
        }
        adjacencies = {
            f"rank_{r}": connectivity_matrices[f"hodge_laplacian_{r}"]
            for r in range(self.sccn_backbone.layers[0].max_rank + 1)
        }
        output = self.sccn_backbone(features, incidences, adjacencies)
        if len(output) == 3:
            x_0, x_1, x_2 = (
                output["rank_0"],
                output["rank_1"],
                output["rank_2"],
            )
            out_0 = self.readout_0(x_0, x_bel_0)
            out_1 = self.readout_1(x_1, x_bel_1)
            out_2 = self.readout_2(x_2, x_bel_2)
            out = out_0 + out_1 + out_2

        elif len(output) == 2:
            x_0, x_1 = output["rank_0"], output["rank_1"]
            out_0 = self.readout_0(x_0, x_bel_0)
            out_1 = self.readout_1(x_1, x_bel_1)
            out = out_0 + out_1
        else:
            raise ValueError(
                f"Invalid number of output tensors: {len(output)}"
            )
        return out
