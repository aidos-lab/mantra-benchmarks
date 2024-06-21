import torch
from torch import nn

from .readouts.SumReadout import SumReadout
from ..model_types import ModelType
from pydantic import BaseModel
from topomodelx.nn.simplicial.scn2 import SCN2


class SCNConfig(BaseModel):
    type: ModelType = ModelType.SCN
    in_channels_0: int
    in_channels_1: int
    in_channels_2: int
    out_channels: int
    n_layers: int = 2


def normalize_matrix_scn(matrix):
    # Extracted from TopoBenchmarkX
    r"""Normalize the input matrix.

    The normalization is performed using the diagonal matrix of the inverse square root of the sum of the absolute values of the rows.

    Parameters
    ----------
    matrix : torch.sparse.FloatTensor
        Input matrix to be normalized.

    Returns
    -------
    torch.sparse.FloatTensor
        Normalized matrix.
    """
    matrix_ = matrix.to_dense()
    n, _ = matrix_.shape
    abs_matrix = abs(matrix_)
    diag_sum = abs_matrix.sum(axis=1)

    # Handle division by zero
    idxs = torch.where(diag_sum != 0)
    diag_sum[idxs] = 1.0 / torch.sqrt(diag_sum[idxs])

    diag_indices = torch.stack([torch.arange(n), torch.arange(n)])
    diag_matrix = torch.sparse_coo_tensor(
        diag_indices, diag_sum, matrix_.shape, device=matrix.device
    ).coalesce()
    normalized_matrix = diag_matrix @ (matrix @ diag_matrix)
    return normalized_matrix


class SCN(nn.Module):
    """Simplex Convolutional Network Implementation.

        Original paper: Simplicial Complex Neural Networks (https://ieeexplore.ieee.org/document/10285604)


        Parameters
        ----------
        in_channels_0 : int
            Dimension of input features on nodes.
        in_channels_1 : int
            Dimension of input features on edges.
        in_channels_2 : int
            Dimension of input features on faces.
        out_channels : int
            Dimension of output features.
        n_layers : int
            Amount of message passing layers.

        """

    def __init__(self, config: SCNConfig):
        super().__init__()
        self.scn_backbone = SCN2(
            config.in_channels_0,
            config.in_channels_1,
            config.in_channels_2,
            config.n_layers,
        )
        self.readout_0 = SumReadout(config.in_channels_0, config.out_channels)
        self.readout_1 = SumReadout(config.in_channels_1, config.out_channels)
        self.readout_2 = SumReadout(config.in_channels_2, config.out_channels)

    def forward(self, batch):
        x = batch.x
        connectivity_matrices = batch.connectivity
        x_belonging = batch.x_belonging
        x_0, x_1, x_2 = x[0], x[1], x[2]
        x_bel_0, x_bel_1, x_bel_2 = x_belonging[0], x_belonging[1], x_belonging[2]
        laplacian_0 = normalize_matrix_scn(connectivity_matrices['hodge_laplacian_0'])
        laplacian_1 = normalize_matrix_scn(connectivity_matrices['hodge_laplacian_1'])
        laplacian_2 = normalize_matrix_scn(connectivity_matrices['hodge_laplacian_2'])
        x_0, x_1, x_2 = self.scn_backbone(x_0, x_1, x_2, laplacian_0, laplacian_1, laplacian_2)
        out_0 = self.readout_0(x_0, x_bel_0)
        out_1 = self.readout_1(x_1, x_bel_1)
        out_2 = self.readout_2(x_2, x_bel_2)
        return out_0 + out_1 + out_2




