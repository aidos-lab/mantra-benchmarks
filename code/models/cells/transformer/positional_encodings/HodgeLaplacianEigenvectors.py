import numpy as np
import torch
from jaxtyping import Float
from toponetx import CellComplex, SimplicialComplex

from math_utils import (
    compute_hodge_laplacian_matrix,
    normalize_laplacian,
    eigenvectors_smallest_k_eigenvalues,
)
from models.cells.transformer.positional_encodings.BasePositionalEncodings import (
    BasePositionalEncodings,
)


def _hodge_laplacian_positional_encoding_precomputed(
    t_complex: SimplicialComplex | CellComplex,
    dim_positional_encodings: int,
    length_pos_enc: int,
    padding=False,
    normalize=False,
) -> np.ndarray:
    H = compute_hodge_laplacian_matrix(
        t_complex, dim_positional_encodings, signed=True
    )
    n = H.shape[0]
    if normalize:
        H = normalize_laplacian(H)
    if not padding and n <= length_pos_enc:
        assert (
            "the number of eigenvectors k must be smaller than the number of "
            + f"nodes n, {length_pos_enc} and {n} detected."
        )
    return eigenvectors_smallest_k_eigenvalues(H, length_pos_enc)


class HodgeLaplacianEigenvectorsPE(BasePositionalEncodings):
    def generate_positional_encodings(
        self,
        t_complex: CellComplex | SimplicialComplex,
        length_positional_encodings: int,
    ) -> dict[int, Float[torch.Tensor, "n_dim length_positional_encodings"]]:
        pe = dict()
        for dim in range(t_complex.dim + 1):
            pes = _hodge_laplacian_positional_encoding_precomputed(
                t_complex,
                dim,
                length_positional_encodings,
                padding=True,
                normalize=False,
            )
            pe[dim] = torch.tensor(pes, dtype=torch.float32)
        return pe
