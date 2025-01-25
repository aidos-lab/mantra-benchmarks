import numpy as np
import torch
from jaxtyping import Float
from toponetx.classes import CellComplex, SimplicialComplex

from CellComplexCombinatorics import lower_adjacency, upper_adjacency
from models.cells.transformer.positional_encodings.BasePositionalEncodings import (
    BasePositionalEncodings,
)


def random_walk_positional_encoding(
    t_complex: CellComplex | SimplicialComplex,
    dim_positional_encodings: int,
    length_pos_enc: int,
):
    if isinstance(t_complex, SimplicialComplex):
        raise NotImplementedError("SimplicialComplex not supported yet.")
    RW = compute_rw_cell_complex(t_complex, dim_positional_encodings)
    return generate_pe_from_transition_matrix(RW, length_pos_enc)


def compute_adjacency_matrices(cell_complex: CellComplex, dim: int):
    adjacency_lower = lower_adjacency(cell_complex, dim, s=1)
    adjacency_upper = upper_adjacency(cell_complex, dim, s=1)
    return adjacency_lower, adjacency_upper


def compute_lower_and_upper_degrees(adjacency_lower, adjacency_upper):
    lower_degrees = np.asarray(adjacency_lower.sum(axis=0)).flatten()
    upper_degrees = np.asarray(adjacency_upper.sum(axis=0)).flatten()
    return lower_degrees, upper_degrees


def compute_rw_cell_complex(cell_complex: CellComplex, dim: int):
    adjacency_lower, adjacency_upper = compute_adjacency_matrices(
        cell_complex, dim
    )
    lower_degrees, upper_degrees = compute_lower_and_upper_degrees(
        adjacency_lower, adjacency_upper
    )
    lower_isolated_cells = np.where(lower_degrees == 0, 1.0, 0.0)
    upper_isolated_cells = np.where(upper_degrees == 0, 1.0, 0.0)
    # If cells are isolated, we need to add a self-loop to the adjacency matrix
    corrected_adjacency_lower = adjacency_lower + np.diag(lower_isolated_cells)
    corrected_adjacency_upper = adjacency_upper + np.diag(upper_isolated_cells)
    # If the original degree of a cell is zero, we added a self-loop to the adjacency matrix and thus the degree is one
    corrected_lower_degrees = np.maximum(lower_degrees, lower_isolated_cells)
    corrected_upper_degrees = np.maximum(upper_degrees, upper_isolated_cells)
    # Compute the random walk matrices
    rw_up = corrected_adjacency_upper @ np.diag(1.0 / corrected_upper_degrees)
    rw_low = corrected_adjacency_lower @ np.diag(1.0 / corrected_lower_degrees)
    # Compute the combined random walk matrix
    rw_combined = np.zeros_like(rw_up)
    for i in range(rw_up.shape[0]):
        for j in range(rw_up.shape[1]):
            if upper_degrees[j] != 0 and lower_degrees[j] != 0:
                rw_combined[i, j] = 0.5 * rw_up[i, j] + 0.5 * rw_low[i, j]
            elif upper_degrees[j] != 0 and lower_degrees[j] == 0:
                rw_combined[i, j] = rw_up[i, j]
            elif upper_degrees[j] == 0 and lower_degrees[j] != 0:
                rw_combined[i, j] = rw_low[i, j]
            else:
                rw_combined[i, j] = 1.0 if i == j else 0.0
    assert np.allclose(
        np.asarray(rw_combined.sum(axis=0)).flatten(), 1.0
    )  # Check columns sum to one
    return np.asarray(rw_combined)


def generate_pe_from_transition_matrix(RW, length_pos_enc: int):
    RW_acc = RW
    diagonals_rw = [RW_acc.diagonal()]
    for i in range(length_pos_enc - 1):
        RW_acc = RW_acc @ RW
        diagonals_rw.append(RW_acc.diagonal())
    random_walk_probs = np.stack(diagonals_rw, axis=1)
    return random_walk_probs


class RandomWalkPE(BasePositionalEncodings):
    def generate_positional_encodings(
        self,
        t_complex: CellComplex | SimplicialComplex,
        length_positional_encodings: int,
    ) -> dict[int, Float[torch.Tensor, "n_dim length_positional_encodings"]]:
        pe = dict()
        for dim in range(t_complex.dim + 1):
            pes = random_walk_positional_encoding(
                t_complex=t_complex,
                dim_positional_encodings=dim,
                length_pos_enc=length_positional_encodings,
            )
            pe[dim] = torch.tensor(pes, dtype=torch.float32)
        return pe
