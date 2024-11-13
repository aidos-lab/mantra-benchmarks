import numpy as np
import scipy
import torch
from dgl.sparse import SparseMatrix
from toponetx import SimplicialComplex
import dgl.sparse as dglsp
from toponetx import SimplicialComplex, CellComplex

from models.cells.CellComplexCombinatorics import hodge_laplacian_matrix


def barycentric_subdivision(K: SimplicialComplex) -> SimplicialComplex:
    # Create a new SimplicialComplex to store the subdivision
    Sd_K = SimplicialComplex()

    new_simplices = {dim: set() for dim in range(K.dim + 1)}

    # Add new vertices to Sd_K. Each simplex of Sd_K is a chain of simplices of K
    for simplex in K.simplices:
        new_simplices[0].add((simplex,))

    # Give now an index to each simplex
    simplex_to_index = {
        simplex[0]: i for i, simplex in enumerate(new_simplices[0])
    }

    # Now, we add simplices from dimension 1 to K.dim
    for dim in range(1, K.dim + 1):
        # Get all simplices of the previous dimension, and try to add more simplices to the chain
        previous_simplices = new_simplices[dim - 1]
        for simplex_sub in previous_simplices:
            last_simplex = simplex_sub[-1]
            for simplex in K.simplices:
                # Check if simplex is a face of simplex_sub
                if last_simplex < simplex:
                    new_simplices[dim].add(simplex_sub + (simplex,))
    # Now convert the simplices to indexes
    all_simplices = []
    for dim in range(K.dim + 1):
        for simplex in new_simplices[dim]:
            all_simplices.append(
                [simplex_to_index[or_simplex] for or_simplex in simplex]
            )
    # Add the simplices to the new SimplicialComplex
    Sd_K.add_simplices_from(all_simplices)
    return Sd_K, simplex_to_index


def recursive_barycentric_subdivision(
        K: SimplicialComplex, number_of_transformations: int
) -> SimplicialComplex:
    Sd_K = K
    for _ in range(number_of_transformations):
        Sd_K, _ = barycentric_subdivision(Sd_K)
    return Sd_K


def sparse_abs(M: SparseMatrix) -> SparseMatrix:
    return dglsp.val_like(M, torch.abs(M.val))

def eigenvectors_smallest_k_eigenvalues(L, k):
    n = L.shape[0]
    # select eigenvectors with smaller eigenvalues O(n + klogk)
    EigVal, EigVec = np.linalg.eig(L.toarray())
    max_freqs = min(n - 1, k)
    kpartition_indices = np.argpartition(EigVal, max_freqs)[: max_freqs + 1]
    topk_eigvals = EigVal[kpartition_indices]
    topk_indices = kpartition_indices[topk_eigvals.argsort()][1:]
    topk_EigVec = EigVec[:, topk_indices]
    # get random flip signs
    rand_sign = 2 * (np.random.rand(max_freqs) > 0.5) - 1.0
    PE = rand_sign * topk_EigVec
    # add paddings
    if n <= k:
        temp_EigVec = np.zeros((n, k - n + 1), dtype=np.float32)
        PE = np.concatenate((PE, temp_EigVec), axis=1)
    return PE

def compute_hodge_laplacian_matrix(
        t_complex: SimplicialComplex | CellComplex, dim: int, signed=True
):
    # Check type of t_complex
    if isinstance(t_complex, SimplicialComplex):
        return t_complex.hodge_laplacian_matrix(dim, signed=signed)
    elif isinstance(t_complex, CellComplex):
        return hodge_laplacian_matrix(t_complex, dim, signed=signed)


def normalize_laplacian(H: scipy.sparse.spmatrix):
    # We assume that the matrix is a scipy sparse matrix. We normalize by multiplying by D^{-1/2} where D is the
    # diagonal matrix of the Hodge Laplacian H.
    inverse_squared_diagonal = (scipy.sparse.diags(H.diagonal()).sqrt()).power(-1)
    return inverse_squared_diagonal @ H @ inverse_squared_diagonal