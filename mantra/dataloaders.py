from typing import Any, Optional

import numpy as np
import scipy
import torch.utils.data

from torch.utils.data.dataloader import DataLoader


class BatchedSimplicialComplex:
    def __init__(
        self,
        signals: dict[Any, torch.Tensor],
        neighborhood_matrices: [
            None | dict[Any, scipy.sparse.spmatrix]
        ] = None,
        other_features: [None | dict[Any, torch.Tensor]] = None,
    ):
        self.signals = signals
        self.neighborhood_matrices = neighborhood_matrices
        self.other_features = other_features


def batch_signals(signals):
    return torch.cat(signals, dim=0)


def convert_sparse_matrices_to_sparse_block_matrix(
    key: Any,
    sparse_matrices: list[Optional[scipy.sparse.spmatrix]],
    batch: list[BatchedSimplicialComplex],
):
    rows, columns, values = [], [], []
    idx_rows, idx_cols = 0, 0
    for matrix, c_complex in zip(sparse_matrices, batch):
        if matrix is None:
            # Depending on the neighborhood matrix, we need to perform different operations
            match key:
                case (
                    "0_adjacency"
                    | "1_coadjacency"
                    | "1_adjacency"
                    | "2_coadjacency"
                    | "1_laplacian"
                    | "2_laplacian"
                    | "0_laplacian"
                    | "1_laplacian_up"
                    | "1_laplacian_down"
                ):
                    # We do not do nothing, as if the matrix is None, it is because there are no cells
                    # of that dimension in the cell complex so we do not need to add any row or column
                    pass
                case "1_boundary":
                    # Only one possibility: the dimension of the cell complex is 0, and therefore we need to add
                    # only as many rows as nodes there are in the cell complex
                    zero_cells_cardinality = len(c_complex.signals[0])
                    idx_rows += zero_cells_cardinality
                case "2_boundary":
                    # Only two possibilities: the dimension of the cell complex is 0 or 1. If the dimension is 0,
                    # we do not add any row or column. If the dimension is 1, we need to add as many rows as edges
                    # there are in the cell complex
                    if 1 not in c_complex.signals:
                        pass
                    else:
                        one_cells_cardinality = len(c_complex.signals[1])
                        idx_rows += one_cells_cardinality
        else:
            coo_matrix = matrix.tocoo()
            len_rows, len_cols = coo_matrix.shape
            rows_example, cols_example, values_example = (
                coo_matrix.row,
                coo_matrix.col,
                coo_matrix.data,
            )
            rows_abs, cols_abs = (
                rows_example + idx_rows,
                cols_example + idx_cols,
            )
            rows.append(rows_abs)
            columns.append(cols_abs)
            values.append(values_example)
            idx_rows += len_rows
            idx_cols += len_cols
    rows_cat = np.concatenate(rows, axis=0)
    columns_cat = np.concatenate(columns, axis=0)
    values_cat = np.concatenate(values, axis=0)
    return scipy.sparse.coo_matrix(
        (values_cat, (rows_cat, columns_cat)), shape=(idx_rows, idx_cols)
    )


def collate_signals(batch):
    all_signals_keys = set(
        [key for example in batch for key in example.signals]
    )
    signals = dict()
    signals_belonging = dict()
    for key in all_signals_keys:
        signals_to_batch = [
            example.signals[key] for example in batch if key in example.signals
        ]
        signals[key] = batch_signals(signals_to_batch)
        signals_of_dim_belonging = [
            torch.tensor([i] * len(batch[i].signals[key]), dtype=torch.int64)
            for i in range(len(batch))
            if key in batch[i].signals
        ]
        signals_belonging[key] = torch.cat(signals_of_dim_belonging, dim=0)
    return signals, signals_belonging


def collate_other_features(batch):
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
            other_features[key] = torch.cat(
                [
                    example.other_features[key]
                    for example in batch
                    if key in example.other_features
                ],
                dim=0,
            )
    return other_features


def collate_neighborhood_matrices(batch):
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
        neighborhood_matrices[key] = (
            convert_sparse_matrices_to_sparse_block_matrix(
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
        )
    return neighborhood_matrices


def generate_batched_simplicial_complex_from_data(data):
    return BatchedSimplicialComplex(
        signals=data.x,
        neighborhood_matrices=data.neighborhood_matrices,
        other_features=data.other_features,
    )


def collate(batch):
    batch = [
        generate_batched_simplicial_complex_from_data(example)
        for example in batch
    ]
    # Collate the signals and make a belonging vector
    signals, signals_belonging = collate_signals(batch)
    # Concatenate other features. Take all the feature names from all the examples in the batch.
    other_features = collate_other_features(batch)
    # Concatenate neighborhood matrices
    neighborhood_matrices = collate_neighborhood_matrices(batch)

    return (
        BatchedSimplicialComplex(
            signals,
            neighborhood_matrices=neighborhood_matrices,
            other_features=other_features,
        ),
        signals_belonging,
        len(batch),
    )


class SimplicialDataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        collate_fn = kwargs.get("collate_fn", collate)
        kwargs = {k: v for k, v in kwargs.items() if k != "collate_fn"}
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)
