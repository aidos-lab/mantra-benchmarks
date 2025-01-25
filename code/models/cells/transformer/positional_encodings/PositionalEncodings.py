from enum import Enum

import torch
from jaxtyping import Float
from toponetx import SimplicialComplex, CellComplex

from models.cells.transformer.positional_encodings.BarycentricSubdivision import (
    BarycentricSubdivisionRandomWalkPE,
    BarycentricSubdivisionEigenvectorsPE,
)
from models.cells.transformer.positional_encodings.HodgeLaplacianEigenvectors import (
    HodgeLaplacianEigenvectorsPE,
)
from models.cells.transformer.positional_encodings.RandomWalks import (
    RandomWalkPE,
)
from models.cells.transformer.positional_encodings.ZeroPositionalEncodings import (
    ZeroPE,
)


class PositionalEncodingsType(Enum):
    HODGE_LAPLACIAN_EIGENVECTORS = "HodgeLapEig"
    ZEROS = "zeros"
    RANDOM_WALKS = "RWPe"
    BARYCENTRIC_SUBDIVISION_GRAPH_EIGENVECTORS = "BSPe"
    BARYCENTRIC_SUBDIVISION_RANDOM_WALKS_GRAPH = "RWBSPe"


def get_positional_encodings(
    t_complex: CellComplex | SimplicialComplex,
    pe_type: PositionalEncodingsType,
    length_positional_encodings: int,
) -> dict[int, Float[torch.Tensor, "n_dim length_positional_encodings"]]:
    match pe_type:
        case PositionalEncodingsType.HODGE_LAPLACIAN_EIGENVECTORS:
            pe_builder = HodgeLaplacianEigenvectorsPE()
        case PositionalEncodingsType.ZEROS:
            pe_builder = ZeroPE()
        case PositionalEncodingsType.RANDOM_WALKS:
            pe_builder = RandomWalkPE()
        case (
            PositionalEncodingsType.BARYCENTRIC_SUBDIVISION_RANDOM_WALKS_GRAPH
        ):
            pe_builder = BarycentricSubdivisionRandomWalkPE()
        case (
            PositionalEncodingsType.BARYCENTRIC_SUBDIVISION_GRAPH_EIGENVECTORS
        ):
            pe_builder = BarycentricSubdivisionEigenvectorsPE()
        case _:
            raise ValueError("Non-recognized positional encoding type")
    return pe_builder.generate_positional_encodings(
        t_complex=t_complex,
        length_positional_encodings=length_positional_encodings,
    )
