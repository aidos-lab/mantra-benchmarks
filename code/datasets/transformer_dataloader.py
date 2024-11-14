from typing import Sequence, Any, Optional

import numpy as np
import scipy
import torch
from dgl.sparse import SparseMatrix
from jaxtyping import Float, Int
from torch.utils.data import DataLoader

from datasets.utils import concat_tensors
from models.cells.transformer.DataTypes import CellComplexData, NeighborhoodMatrixType, NeighborhoodType
from datasets.utils import torch_sparse_to_scipy_sparse


class TransformerDataloader(DataLoader):
    def __init__(self, dataset, **kwargs):
        collate_fn = kwargs.get("collate_fn", collate_transformer_models)
        kwargs = {k: v for k, v in kwargs.items() if k != "collate_fn"}
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)


def sc_to_cell_complex_data(data):
    signals = data.x
    neighborhood_matrices = dict()
    for dim in signals.keys():
        if f'boundary_{dim}' in data.connectivity and dim > 0:
            neighb_type = NeighborhoodMatrixType(NeighborhoodType.BOUNDARY, dim)
            neighborhood_matrices[neighb_type] = torch_sparse_to_scipy_sparse(data.connectivity[f'boundary_{dim}'])
        if f'adjacency_{dim}' in data.connectivity:
            neighb_type = NeighborhoodMatrixType(NeighborhoodType.UPPER_ADJACENCY, dim)
            neighborhood_matrices[neighb_type] = torch_sparse_to_scipy_sparse(data.connectivity[f'adjacency_{dim}'])
        if f'coadjacency_{dim}' in data.connectivity:
            neighb_type = NeighborhoodMatrixType(NeighborhoodType.LOWER_ADJACENCY, dim)
            neighborhood_matrices[neighb_type] = torch_sparse_to_scipy_sparse(data.connectivity[f'coadjacency_{dim}'])
        if f'up_laplacian_{dim}' in data.connectivity:
            neighb_type = NeighborhoodMatrixType(NeighborhoodType.UPPER_HODGE_LAPLACIAN, dim)
            neighborhood_matrices[neighb_type] = torch_sparse_to_scipy_sparse(data.connectivity[f'up_laplacian_{dim}'])
        if f'down_laplacian_{dim}' in data.connectivity:
            neighb_type = NeighborhoodMatrixType(NeighborhoodType.LOWER_HODGE_LAPLACIAN, dim)
            neighborhood_matrices[neighb_type] = torch_sparse_to_scipy_sparse(data.connectivity[f'down_laplacian_{dim}'])
        if f'hodge_{dim}' in data.connectivity:
            neighb_type = NeighborhoodMatrixType(NeighborhoodType.HODGE_LAPLACIAN, dim)
            neighborhood_matrices[neighb_type] = torch_sparse_to_scipy_sparse(data.connectivity[f'hodge_{dim}'])
    other_features = dict()
    other_features["positional_encodings"] = data.pe
    if "y" in other_features:
        other_features["y"] = data.other_features["y"]
    return CellComplexData(signals=signals, neighborhood_matrices=neighborhood_matrices, other_features=other_features)


def collate_signals(
        batch: Sequence[CellComplexData],
) -> tuple[dict[int, Float[torch.Tensor, "..."]], dict[int, Int[torch.Tensor, "..."]]]:
    all_signals_keys = set([key for example in batch for key in example.signals])
    signals = dict()
    signals_belonging = dict()
    for key in all_signals_keys:
        signals_to_batch = [
            example.signals[key] for example in batch if key in example.signals
        ]
        signals[key] = torch.cat(signals_to_batch, dim=0)
        signals_of_dim_belonging = [
            torch.tensor([i] * len(batch[i].signals[key]), dtype=torch.int64)
            for i in range(len(batch))
            if key in batch[i].signals
        ]
        signals_belonging[key] = torch.cat(signals_of_dim_belonging, dim=0)
    return signals, signals_belonging


def collate_other_features_type_dependent(
        feature_list: list[
            Float[torch.Tensor, "..."]
            | dict[Any, Float[torch.Tensor, "..."]]
            | list[Float[torch.Tensor, "..."]]
            ]
) -> (
        Float[torch.Tensor, "..."]
        | dict[Any, Float[torch.Tensor, "..."]]
        | list[Float[torch.Tensor, "..."]]
):
    data_example = feature_list[0]
    collated_subfeatures = None
    if isinstance(data_example, dict):
        collated_subfeatures = dict()
        feature_names = set([key for example in feature_list for key in example.keys()])
        for key in feature_names:
            collated_subfeatures[key] = torch.cat(
                [example[key] for example in feature_list if key in example], dim=0
            )
    if isinstance(data_example, torch.Tensor):
        collated_subfeatures = concat_tensors(feature_list, dim=0)
    if isinstance(data_example, list):
        # We assume that all the lists have the same length.
        collated_subfeatures = [
            torch.cat([example[i] for example in feature_list], dim=0)
            for i in range(len(data_example))
        ]
    if collated_subfeatures is None:
        raise ValueError("Unknown other features type")
    return collated_subfeatures


def collate_other_features(
        batch: Sequence[CellComplexData],
) -> dict[Any, Float[torch.Tensor, "..."]]:
    feature_names = set(
        [
            key
            for example in batch
            if example.other_features is not None
            for key in example.other_features.keys()
        ]
    )
    if len(feature_names) == 0:
        other_features = None
    else:
        other_features = {}
        for key in feature_names:
            other_features[key] = collate_other_features_type_dependent(
                [
                    example.other_features[key]
                    for example in batch
                    if key in example.other_features
                ]
            )
    return other_features


def collate_neighborhood_matrices(
        batch: Sequence[CellComplexData],
) -> dict[NeighborhoodMatrixType, scipy.sparse.spmatrix | SparseMatrix]:
    all_neighborhood_matrices_keys = set(
        [
            key
            for example in batch
            if example.neighborhood_matrices is not None
            for key in example.neighborhood_matrices
        ]
    )
    neighborhood_matrices = dict()
    for key in all_neighborhood_matrices_keys:
        neighborhood_matrices[key] = convert_sparse_matrices_to_sparse_block_matrix(
            key,
            [
                #  Get the neighborhood matrix if it exists, otherwise None
                (
                    example.neighborhood_matrices[key]
                    if key in example.neighborhood_matrices
                    else None
                )
                for example in batch
            ],
            batch,
        )
    return neighborhood_matrices


def convert_sparse_matrices_to_sparse_block_matrix(
        key: NeighborhoodMatrixType,
        sparse_matrices: list[Optional[scipy.sparse.spmatrix | SparseMatrix]],
        batch: list[CellComplexData],
) -> scipy.sparse.spmatrix:
    rows, columns, values = [], [], []
    idx_rows, idx_cols = 0, 0
    for matrix, c_complex in zip(sparse_matrices, batch):
        # Matrix is None implies that the matrix does not exist in the associated cell complex of the batch. This means
        # that the cell complex does not have cells at least of one of the dimensions associated with the matrix.
        # In this case, for matrices going from dimension n to n-1 we need to add as many rows as there are cells of
        # dimension n-1 in the cell complex, if any. In the case of matrices going from dimension n to dimension n,
        # we do not need to add any row or column.
        if isinstance(matrix, SparseMatrix):
            raise ValueError(
                "DGL SparseMatrix is not supported for collating. Use scipy.sparse.spmatrix instead."
            )
        if matrix is None:
            # Depending on the neighborhood matrix, we need to perform different operations
            match key.type:
                case (
                NeighborhoodType.UPPER_ADJACENCY
                | NeighborhoodType.LOWER_ADJACENCY
                | NeighborhoodType.LOWER_HODGE_LAPLACIAN
                | NeighborhoodType.UPPER_HODGE_LAPLACIAN
                | NeighborhoodType.HODGE_LAPLACIAN
                ):
                    # In this case we have two possibilities.
                    # (1) There are no cells of that dimension in the cell complex, so we do not need to add any
                    # row or column. We do nothing.
                    # (2) There are cells of that dimension in the cell complex, but the matrix is not well defined
                    # (i.e., in dimension zero one could argue that you cannot have a lower adjacency matrix or lower
                    # hodge laplacian matrix). In this case, we should add as many rows and columns as there are cells
                    # of that dimension in the cell complex.
                    if c_complex.dim >= key.dimension:
                        c_dim_cardinality = len(c_complex.signals[key.dimension])
                        idx_rows += c_dim_cardinality
                        idx_cols += c_dim_cardinality
                case NeighborhoodType.BOUNDARY:
                    # Boundary dimensions are always greater or equal than one. There are two possibilities.
                    # The dimension of the cell complex c is equal to the boundary dimension minus one, meaning that
                    # we should add as many rows as there are cells of dimension c in the cell complex. The other
                    # possibility is that c is lower than the boundary dimension minus one, meaning that we should
                    # add no rows or columns.

                    if c_complex.dim == key.dimension - 1:
                        c_dim_cardinality = len(c_complex.signals[key.dimension - 1])
                        idx_rows += c_dim_cardinality
                    else:
                        pass
                case _:
                    raise ValueError(f"Unknown neighborhood matrix type {key.type}")
        else:
            coo_matrix = matrix.tocoo()
            len_rows, len_cols = coo_matrix.shape
            rows_example, cols_example, values_example = (
                coo_matrix.row,
                coo_matrix.col,
                coo_matrix.data,
            )
            rows_abs, cols_abs = rows_example + idx_rows, cols_example + idx_cols
            rows.append(rows_abs)
            columns.append(cols_abs)
            values.append(values_example)
            idx_rows += len_rows
            idx_cols += len_cols
    rows_cat = np.concatenate(rows, axis=0)
    columns_cat = np.concatenate(columns, axis=0)
    values_cat = np.concatenate(values, axis=0)
    if idx_rows != sum(
            [
                example.signals[key.dimension - 1].shape[0]
                for example in batch
                if (key.dimension - 1) in example.signals
            ]
    ) and idx_rows != sum(
        [
            example.signals[key.dimension].shape[0]
            for example in batch
            if key.dimension in example.signals
        ]
    ):
        print(f"Error for key {key}")
    return scipy.sparse.coo_matrix(
        (values_cat, (rows_cat, columns_cat)), shape=(idx_rows, idx_cols)
    )


def collate_cell_complex(batch: list[CellComplexData]) -> CellComplexData:
    # Collate the signals and make a belonging vector
    signals, signals_belonging = collate_signals(batch)
    # Concatenate other features. Take all the feature names from all the examples in the batch.
    other_features = collate_other_features(batch)
    # Concatenate neighborhood matrices
    neighborhood_matrices = collate_neighborhood_matrices(batch)
    # Add signals_belonging to other_features.
    other_features["signals_belonging"] = signals_belonging
    return CellComplexData(
        signals=signals,
        neighborhood_matrices=neighborhood_matrices,
        other_features=other_features,
        batch_size=len(batch),
    )


def collate_transformer_models(batch):
    # First we transform all sc to cell complex format
    batch_formated = [sc_to_cell_complex_data(data) for data in batch]
    return collate_cell_complex(batch_formated)
