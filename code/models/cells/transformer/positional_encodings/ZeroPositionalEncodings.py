import torch
from jaxtyping import Float
from toponetx import CellComplex, SimplicialComplex

from models.cells.transformer.positional_encodings.BasePositionalEncodings import (
    BasePositionalEncodings,
)


class ZeroPE(BasePositionalEncodings):
    def generate_positional_encodings(
        self,
        t_complex: CellComplex | SimplicialComplex,
        length_positional_encodings: int,
    ) -> dict[int, Float[torch.Tensor, "n_dim length_positional_encodings"]]:
        pe = dict()
        for dim in range(t_complex.dim + 1):
            pe[dim] = torch.zeros(
                (t_complex.shape[dim], length_positional_encodings)
            )
        return pe
