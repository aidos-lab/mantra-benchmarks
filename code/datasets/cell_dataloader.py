import numpy as np
import torch
from toponetx import SimplicialComplex
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from typing import List
from datasets.utils import eliminate_zeros_torch_sparse, torch_sparse_to_scipy_sparse
from models.cells.mp.complex import ComplexBatch, Cochain, CochainBatch, Complex


class CellDataloader(DataLoader):
    def __init__(self, dataset, **kwargs):
        collate_fn = kwargs.get("collate_fn", collate_cell_models)
        kwargs = {k: v for k, v in kwargs.items() if k != "collate_fn"}
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)


def get_boundary_index(sc: Data, dim: int):
    if dim == 0:
        return None  # No boundary for vertices


def _get_shared_simplices(data: Data, adj_index, dim: int, cofaces: bool = False):
    simplices_dim = {i: simplex for i, simplex in enumerate(data.sc.skeleton(dim))}
    simplices_related_dim = dim + 1 if cofaces else dim - 1
    simplices_related = {simplex: i for i, simplex in enumerate(data.sc.skeleton(simplices_related_dim))}
    common_cofaces = []
    for i in range(adj_index.shape[1]):
        s1_idx = adj_index[0, i].item()
        s2_idx = adj_index[1, i].item()
        s1 = simplices_dim[s1_idx]
        s2 = simplices_dim[s2_idx]
        if cofaces:
            common_simplex = tuple(sorted(frozenset(s1).union(s2)))
        else:
            common_simplex = tuple(sorted(frozenset(s1).intersection(s2)))
        common_simplex_idx = simplices_related[common_simplex]
        common_cofaces.append(common_simplex_idx)
    return torch.tensor(common_cofaces)


from scipy import sparse


def extract_adj_from_boundary(B, G=None):
    A = sparse.csr_matrix(B.T).dot(sparse.csr_matrix(B))

    n = A.shape[0]
    if G is not None:
        assert n == G.number_of_edges()

    # Subtract self-loops, which we do not count.
    connections = A.count_nonzero() - np.sum(A.diagonal() != 0)

    index = torch.empty((2, connections), dtype=torch.long)
    orient = torch.empty(connections)

    connection = 0
    cA = A.tocoo()
    for i, j, v in zip(cA.row, cA.col, cA.data):
        if j >= i:
            continue
        assert v == 1 or v == -1, print(v)

        index[0, connection] = i
        index[1, connection] = j
        orient[connection] = float(np.sign(v))

        index[0, connection + 1] = j
        index[1, connection + 1] = i
        orient[connection + 1] = float(np.sign(v))

        connection += 2

    assert connection == connections
    return index, orient


def data_to_cochain(data: Data, dim: int, max_dim: int):
    """
    Convert a SimplicialComplex object into a Cochain object for a given dimension.
    """
    x = data.x[dim]
    upper_index, upper_orient = extract_adj_from_boundary(
        torch_sparse_to_scipy_sparse(data.connectivity[f"boundary_{dim+1}"].T)
    ) if (dim < data.sc.dim) and (dim < max_dim) else (None, None)
    lower_index, lower_orient = extract_adj_from_boundary(
        torch_sparse_to_scipy_sparse(data.connectivity[f"boundary_{dim}"])
    ) if (dim < data.sc.dim) and (dim > 0) else (None, None)
    shared_boundaries = _get_shared_simplices(data, lower_index, dim, cofaces=False) \
        if lower_index is not None else None
    shared_coboundaries = _get_shared_simplices(data, upper_index, dim, cofaces=True) \
        if upper_index is not None else None
    boundary_index = data.connectivity[f"boundary_{dim}"].indices() if dim > 0 else None
    y = getattr(data, "y", None)
    # TODO: Mapping is not used in their implementation, so I leave it as None for now
    return Cochain(dim, x, upper_index, lower_index, shared_boundaries, shared_coboundaries,
                   None, boundary_index, upper_orient, lower_orient, y)


def collate_cell_models(batch: List[Data]) -> ComplexBatch:
    """
    Convert a list of SimplicialComplex objects into a ComplexBatch.

    Data Point 1: [Cochain(0D), Cochain(1D), Cochain(2D)] → Complex(Data Point 1)
    Data Point 2: [Cochain(0D), Cochain(1D), Cochain(2D)] → Complex(Data Point 2)

    ComplexBatch = [Complex(Data Point 1), Complex(Data Point 2)]
    """
    complexes = []
    max_dim = max(x.sc.dim for x in batch)

    for data in batch:
        cochains = []
        dim = max_dim
        for dim in range(max_dim + 1):
            cochain = data_to_cochain(data, dim, max_dim)
            cochains.append(cochain)
        complex = Complex(
            *cochains,
            y=data.y,
            dimension=max_dim
        )
        complexes.append(complex)

    cochain_batch = ComplexBatch.from_complex_list(
        data_list=complexes,
        max_dim=max_dim
    )

    return cochain_batch
