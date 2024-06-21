import torch
from toponetx import SimplicialComplex
from torch.utils.data import DataLoader
from torch_geometric.data import Data


class SimplicialTopoXDataloader(DataLoader):
    def __init__(self, dataset, **kwargs):
        collate_fn = kwargs.get("collate_fn", collate_simplicial_models_topox)
        kwargs = {k: v for k, v in kwargs.items() if k != "collate_fn"}
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)


def batch_connectivity_matrices(key, matrices, batch):
    rows, columns, values = [], [], []
    matrix_dim = int(key.split("_")[-1])
    row_idx, col_idx = 0, 0
    for matrix, example in zip(matrices, batch):
        if matrix is None:
            if key.startswith("incidence"):
                # We only add rows if the simplicial complex has simplices of dimension matrix_dim - 1.
                # because otherwise we do not have simplices of the dimensions represented by the rows and columns.
                if example.dim == matrix_dim - 1:
                    # The simplicial complex has a non-empty set of simplices of dimension matrix_dim - 1,
                    # but it does not have simplices of dimension matrix_dim
                    row_idx += example.shape(matrix_dim - 1)
            elif (key.startswith("down_laplacian") or key.startswith("up_laplacian")
                  or key.startswith("hodge_laplacian")) or key.startswith("adjacency"):
                # We do not do nothing, as if the matrix is None, it is because there are no cells
                # of that dimension in the cell complex so we do not need to add any row or column
                pass
            else:
                raise NotImplementedError(f"{key} is not valid connectivity matrix.")
        else:
            indices = matrix.indices()
            rows_submatrix = indices[0]
            cols_submatrix = indices[1]
            rows.append(rows_submatrix + row_idx)
            columns.append(cols_submatrix + col_idx)
            values.append(matrix.values())
            row_idx += matrix.shape[0]
            col_idx += matrix.shape[1]
    rows_cat = torch.cat(rows, dim=0)
    columns_cat = torch.cat(columns, dim=0)
    values_cat = torch.cat(values, dim=0)
    return torch.sparse_coo_tensor(torch.stack([rows_cat, columns_cat]), values_cat, (row_idx, col_idx))


def collate_connectivity_matrices(batch):
    connectivity_batched = dict()
    connectivity_keys = set([key
                             for example in batch
                             if example.connectivity is not None
                             for key in example.connectivity.keys()])
    for key in connectivity_keys:
        connectivity_batched[key] = batch_connectivity_matrices(key,
                                                                [example.connectivity[key]
                                                                 if key in example.connectivity
                                                                 else None
                                                                 for example in batch],
                                                                batch)
    return connectivity_batched


def collate_signals(batch):
    x_batched = dict()
    all_x_keys = set([key for example in batch for key in example.x.keys()])
    x_belonging = dict()
    for key in all_x_keys:
        x_to_batch = [
            example.x[key] for example in batch if key in example.x
        ]
        x_batched[key] = torch.cat(x_to_batch, dim=0)
        signals_of_dim_belonging = [
            torch.tensor([i] * len(batch[i].x[key]), dtype=torch.int64)
            for i in range(len(batch))
            if key in batch[i].x
        ]
        x_belonging[key] = torch.cat(signals_of_dim_belonging, dim=0)
    return x_batched, x_belonging


def collate_simplicial_models_topox(batch):
    batched_data = Data()
    batched_data.batch_size = len(batch)
    # First, batch signals
    x_batched, x_belonging = collate_signals(batch)
    batched_data.x = x_batched
    batched_data.x_belonging = x_belonging
    # Second, batch structure matrices
    batched_data.connectivity = collate_connectivity_matrices(batch)
    # Third, batch output
    batched_data.y = torch.cat([example.y for example in batch], dim=0)
    return batched_data
