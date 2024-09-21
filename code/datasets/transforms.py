import torch
from toponetx.classes import SimplicialComplex
from torch_geometric.utils import degree
from torch_geometric.transforms import FaceToEdge, OneHotDegree
from datasets.utils import (
    create_signals_on_data_if_needed,
    append_signals,
    get_complex_connectivity,
    create_or_empty_signals_on_data,
    get_triangles_from_simplicial_complex,
)
from enum import Enum
from datasets.dataset_types import DatasetType
from typing import List, Callable


from math_utils import recursive_barycentric_subdivision

NAME_TO_CLASS = {"Klein bottle": 0, "RP^2": 1, "T^2": 2, "S^2": 3, "": 4}


class SetNumNodesTransform:
    def __call__(self, data):
        data.num_nodes = data.n_vertices
        return data


class OrientableToClassTransform:
    def __call__(self, data):
        data.orientable = torch.tensor(data.betti_numbers)[..., -1]
        data.y = data.orientable.long()
        return data


class BettiToY2:
    def __call__(self, data):
        data.y = torch.tensor(data.betti_numbers, dtype=torch.float).view(1, 3)
        return data


class BettiToY3:
    def __call__(self, data):
        data.y = torch.tensor(data.betti_numbers, dtype=torch.float).view(1, 4)
        return data


class DegreeTransform:
    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        data.x = deg.view(-1, 1)
        return data


class TriangulationToZeroIndexTransform:
    def __call__(self, data):
        data.face = torch.tensor(data.triangulation).T - 1
        data.num_nodes = data.face.max() + 1
        data.triangulation = None
        return data


class TriangulationToFaceTransform:
    """
    Transforms tetrahedra to faces.
    Expects a triangulation of shape [4,N] and
    returns the faces of shape [3,M].

    NOTE: It will contain duplicate triangles with different ordering.
    Hence the result is not a "minimal" triangulation. When subsequently
    converting the triangles to edges, this will not pose a problem as
    the FaceToEdge transform by default creates undirected edges.
    """

    def __init__(self, remove_triangulation: bool = True) -> None:
        self.remove_triangulation = remove_triangulation

    def __call__(self, data):
        idx = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
        if hasattr(data, "triangulation"):
            assert data.triangulation is not None
            face = torch.cat(
                [torch.tensor(data.triangulation)[i] for i in idx], dim=1
            )

            # Remove duplicate triangles in
            data.face = torch.unique(face, dim=1)

            if self.remove_triangulation:
                data.triangulation = None

        return data


class SimplicialComplexTransform:
    def __call__(self, data):
        data.sc = SimplicialComplex(data.triangulation)
        create_signals_on_data_if_needed(data)
        return data


class BarycentricSubdivisionTransform:
    def __init__(self, recursive_calls=1):
        self.recursive_calls = recursive_calls

    def __call__(self, data):
        data.sc = recursive_barycentric_subdivision(
            data.sc, self.recursive_calls
        )
        data.triangulation = get_triangles_from_simplicial_complex(data)
        data = create_or_empty_signals_on_data(data)
        return data


class SimplicialComplexDegreeTransform:
    def __call__(self, data):
        data = create_signals_on_data_if_needed(data)
        degree_signals = torch.from_numpy(
            data.sc.adjacency_matrix(0).sum(axis=1)
        )
        data = append_signals(data, 0, degree_signals)
        return data


class SimplicialComplexOnesTransform:
    def __init__(self, ones_length=10):
        self.ones_length = ones_length

    def __call__(self, data):
        data = create_signals_on_data_if_needed(data)
        for dim in range(len(data.sc.shape)):
            ones_signals = torch.ones(data.sc.shape[dim], self.ones_length)
            data = append_signals(data, dim, ones_signals)
        return data


class SimplicialComplexEdgeCoadjacencyDegreeTransform:
    def __call__(self, data):
        data = create_signals_on_data_if_needed(data)
        degree_signals = torch.from_numpy(
            data.sc.coadjacency_matrix(1).sum(axis=1)
        )
        data = append_signals(data, 1, degree_signals)
        return data


class SimplicialComplexEdgeAdjacencyDegreeTransform:
    def __call__(self, data):
        data = create_signals_on_data_if_needed(data)
        degree_signals = torch.from_numpy(
            data.sc.adjacency_matrix(1).sum(axis=1)
        )
        data = append_signals(data, 1, degree_signals)
        return data


class SimplicialComplexTriangleCoadjacencyDegreeTransform:
    def __call__(self, data):
        data = create_signals_on_data_if_needed(data)
        degree_signals = torch.from_numpy(
            data.sc.coadjacency_matrix(2).sum(axis=1)
        )
        data = append_signals(data, 2, degree_signals)
        return data


class OrientableToClassSimplicialComplexTransform:
    def __call__(self, data):
        data.other_features["y"] = data.orientable.long()
        return data


class SimplicialComplexStructureMatricesTransform:
    def __call__(self, data):
        data.connectivity = get_complex_connectivity(data.sc, data.sc.dim)
        return data


class NameToClassTransform:
    def __init__(self):
        self.class_dict = NAME_TO_CLASS

    def __call__(self, data):
        # data.y = F.one_hot(torch.tensor(self.class_dict[data.name]),num_classes=5)
        data.y = torch.tensor(self.class_dict[data.name])
        return data


class RandomNodeFeatures:
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        data.x = torch.normal(0, 1, size=(data.num_nodes, self.size))
        return data


class RandomSimplicesFeatures:
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        data = create_signals_on_data_if_needed(data)
        for dim in range(data.sc.dim + 1):
            data = append_signals(
                data,
                dim,
                torch.normal(0, 1, size=(data.sc.shape[dim], self.size)),
            )
        return data


random_node_features = [
    TriangulationToZeroIndexTransform(),
    FaceToEdge(remove_faces=False),
    RandomNodeFeatures(size=8),
]

degree_transform_onehot = [
    TriangulationToZeroIndexTransform(),
    FaceToEdge(remove_faces=False),
    OneHotDegree(max_degree=9, cat=False),
]

degree_transform = [
    TriangulationToZeroIndexTransform(),
    FaceToEdge(remove_faces=False),
    DegreeTransform(),
]

orientability_transforms = [
    OrientableToClassTransform(),
]
name_transforms = [
    NameToClassTransform(),
]
betti_numbers_transforms_2manifold = [
    BettiToY2(),
]

betti_numbers_transforms_3manifold = [
    BettiToY3(),
]

degree_transform_sc = [
    SimplicialComplexTransform(),
    SimplicialComplexStructureMatricesTransform(),
    SimplicialComplexDegreeTransform(),
    SimplicialComplexEdgeAdjacencyDegreeTransform(),
    SimplicialComplexEdgeCoadjacencyDegreeTransform(),
    SimplicialComplexTriangleCoadjacencyDegreeTransform(),
]

random_simplices_features = [
    SimplicialComplexTransform(),
    SimplicialComplexStructureMatricesTransform(),
    RandomSimplicesFeatures(size=8),
]


class TransformType(Enum):
    degree_transform = "degree_transform"
    degree_transform_onehot = "degree_transform_onehot"
    random_node_features = "random_node_features"
    degree_transform_sc = "degree_transform_sc"
    random_simplices_features = "random_simplices_features"


def transforms_lookup(
    tr_type: TransformType, ds_type: DatasetType
) -> List[Callable]:
    _transforms_lookup = {
        TransformType.degree_transform: degree_transform,
        TransformType.degree_transform_onehot: degree_transform_onehot,
        TransformType.random_node_features: random_node_features,
        TransformType.degree_transform_sc: degree_transform_sc,
        TransformType.random_simplices_features: random_simplices_features,
    }

    tr = _transforms_lookup[tr_type]
    if (
        tr_type != TransformType.degree_transform_sc
        and tr_type != TransformType.random_simplices_features
    ):
        tr[0] = TriangulationToFaceTransform()

    return tr
