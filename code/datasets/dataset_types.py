from enum import Enum

from torch_geometric.data import Data


class DatasetType(Enum):
    """
    DESCRIPTION:

    - FULL_2D - Mantra on 2D manifolds
    - FULL_3D - Mantra on 3D manifolds
    - NO_NAMELESS_2D - Mantra on 2D manifolds only including simplicial complexes which have a name label.
    """

    FULL_2D = "full_2d"
    FULL_3D = "full_3d"
    NO_NAMELESS_2D = "no_nameless_2d"


def filter_nameless(data: Data) -> bool:
    return data.name != ""
