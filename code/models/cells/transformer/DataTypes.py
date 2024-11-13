from dataclasses import dataclass
from enum import Enum
from typing import Any

import scipy
import torch
from dgl.sparse import SparseMatrix
from jaxtyping import Float

from utils import (
    scipy_sparse_matrix_to_dgl_sparse,
    dict_of_tensors_to_device,
    list_of_tensors_to_device,
)


class NeighborhoodType(Enum):
    UPPER_ADJACENCY = 0
    LOWER_ADJACENCY = 1
    BOUNDARY = 2
    UPPER_HODGE_LAPLACIAN = 3
    LOWER_HODGE_LAPLACIAN = 4
    HODGE_LAPLACIAN = 5


@dataclass
class NeighborhoodMatrixType:
    type: NeighborhoodType
    dimension: int

    def __hash__(self):
        return hash((self.type, self.dimension))


def _other_features_to_device(
        feature: torch.Tensor | list[torch.Tensor] | dict[Any, torch.Tensor],
        device: torch.device,
) -> torch.Tensor | list[torch.Tensor] | dict[Any, torch.Tensor]:
    if isinstance(feature, torch.Tensor):
        return feature.to(device)
    elif isinstance(feature, list):
        return list_of_tensors_to_device(feature, device)
    else:
        return dict_of_tensors_to_device(feature, device)


@dataclass
class CellComplexData:
    signals: dict[int, Float[torch.Tensor, "..."]]
    neighborhood_matrices: (
            None | dict[NeighborhoodMatrixType, scipy.sparse.spmatrix | SparseMatrix]
    ) = None
    other_features: (
            None
            | dict[
                Any,
                Float[torch.Tensor, "..."]
                | dict[Any, Float[torch.Tensor, "..."]]
                | list[Float[torch.Tensor, "..."]],
            ]
    ) = None
    batch_size: int = 1
    _dim = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = max(self.signals.keys())
        return self._dim

    def to(self, device: torch.device) -> "CellComplexData":
        """
        Moves the data to the specified device. If neighborhood matrices are present and have a scipy.sparse.spmatrix
        format, they are converted to dgl.SparseMatrix format in the correct device.
        :param device: The device to move the data to.
        :return: The data in the specified device.
        """
        # First, we move the signals to the device.
        signals = {key: value.to(device) for key, value in self.signals.items()}
        if self.neighborhood_matrices is not None:
            neighborhood_matrices = {
                key: (
                    value.to(device)
                    if isinstance(value, SparseMatrix)
                    else scipy_sparse_matrix_to_dgl_sparse(value).to(device)
                )
                for key, value in self.neighborhood_matrices.items()
            }
        else:
            neighborhood_matrices = None
        other_features = (
            {
                key: _other_features_to_device(value, device)
                for key, value in self.other_features.items()
            }
            if self.other_features
            else None
        )
        return CellComplexData(
            signals=signals,
            neighborhood_matrices=neighborhood_matrices,
            other_features=other_features,
            batch_size=self.batch_size,
        )


@dataclass
class SimplicialComplexData:
    pass
