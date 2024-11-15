import torch
from toponetx import SimplicialComplex
from torch.utils.data import DataLoader
from torch_geometric.data.data import Data
from typing import List
from models.cells.mp.complex import ComplexBatch


class CellDataloader(DataLoader):
    def __init__(self, dataset, **kwargs):
        collate_fn = kwargs.get("collate_fn", collate_cell_models)
        kwargs = {k: v for k, v in kwargs.items() if k != "collate_fn"}
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)


def simplicial_complex_to_cochain(sc: SimplicialComplex, dim: int, x, y):
    """
    Convert a SimplicialComplex object into a Cochain object for a given dimension.
    """
    raise NotImplementedError()


def collate_cell_models(sc_list: List[Data]) -> ComplexBatch:
    """
    Convert a list of SimplicialComplex objects into a ComplexBatch.
    """
    max_dim = max(x.sc.dim for x in sc_list)
    for dim in range(max_dim + 1):
        pass

    raise NotImplementedError()
