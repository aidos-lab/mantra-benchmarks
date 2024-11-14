from typing import Any

import numpy as np
import scipy
import torch
from dgl import sparse as dglsp
from dgl.sparse import SparseMatrix


def scipy_sparse_matrix_to_dgl_sparse(
    scipy_sparse_matrix: scipy.sparse.spmatrix,
) -> dglsp.SparseMatrix:
    scipy_sparse_matrix = scipy_sparse_matrix.tocoo()
    rows = scipy_sparse_matrix.row
    cols = scipy_sparse_matrix.col
    values = scipy_sparse_matrix.data
    indices = torch.LongTensor(np.vstack([rows, cols]))
    values = torch.FloatTensor(values)
    size = scipy_sparse_matrix.shape
    # If there are no indices or values, we set the element (0, 0) to 0.0, as DGL does not support empty matrices.
    if indices.numel() == 0 and values.numel() == 0:
        indices = torch.LongTensor([[0], [0]])
        values = torch.FloatTensor([0.0])
    return dglsp.spmatrix(indices, values, size)


def dict_of_tensors_to_device(
    tensor_dict: dict[Any, torch.Tensor], device: torch.device
) -> dict[Any, torch.Tensor]:
    return {key: value.to(device) for key, value in tensor_dict.items()}


def list_of_tensors_to_device(
    tensor_list: list[torch.Tensor], device: torch.device
) -> list[torch.Tensor]:
    return [x.to(device) for x in tensor_list]


def generate_repeated_sparse_matrix(
    S: SparseMatrix, times: int
) -> SparseMatrix:
    """
    Allows to generate a sparse matrix with multiple repeated coefficients. Useful
    for summing a sparse matrix to another one with more than one channel.
    :param S: Sparse matrix
    :param times: Number of times to repeat the coefficients
    :return: Sparse matrix with repeated coefficients
    """
    indices = S.indices()
    values = S.val
    new_values = values.repeat_interleave(times).view(-1, times)
    return dglsp.spmatrix(indices, new_values, shape=(S.shape[0], S.shape[1]))
