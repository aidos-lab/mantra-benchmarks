import numpy as np
from torch_geometric.utils import degree
from toponetx.classes import SimplicialComplex
import torch

from mantra.utils import (
    create_signals_on_data_if_needed,
    append_signals,
    create_other_features_on_data_if_needed,
    create_neighborhood_matrices_on_data_if_needed,
)


class OrientableToClassTransform(object):
    def __call__(self, data):
        data.y = data.orientable.long()
        return data


class DegreeTransform(object):
    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        data.x = deg.view(-1, 1)
        return data


class TriangulationToFaceTransform(object):
    def __call__(self, data):
        data.face = torch.tensor(data.triangulation).T - 1
        data.triangulation = None
        return data


class SimplicialComplexTransform(object):
    def __call__(self, data):
        data.sc = SimplicialComplex(data.triangulation)
        create_signals_on_data_if_needed(data)
        create_other_features_on_data_if_needed(data)
        create_neighborhood_matrices_on_data_if_needed(data)
        return data


class SimplicialComplexDegreeTransform(object):
    def __call__(self, data):
        data = create_signals_on_data_if_needed(data)
        degree_signals = torch.from_numpy(
            data.sc.adjacency_matrix(0).sum(axis=1)
        )
        data = append_signals(data, 0, degree_signals)
        return data


class SimplicialComplexOnesTransform(object):
    def __init__(self, ones_length=10):
        self.ones_length = ones_length

    def __call__(self, data):
        data = create_signals_on_data_if_needed(data)
        for dim in range(len(data.sc.shape)):
            ones_signals = torch.ones(data.sc.shape[dim], self.ones_length)
            data = append_signals(data, dim, ones_signals)
        return data


class SimplicialComplexEdgeCoadjacencyDegreeTransform(object):
    def __call__(self, data):
        data = create_signals_on_data_if_needed(data)
        degree_signals = torch.from_numpy(
            data.sc.coadjacency_matrix(1).sum(axis=1)
        )
        data = append_signals(data, 1, degree_signals)
        return data


class SimplicialComplexEdgeAdjacencyDegreeTransform(object):
    def __call__(self, data):
        data = create_signals_on_data_if_needed(data)
        degree_signals = torch.from_numpy(
            data.sc.adjacency_matrix(1).sum(axis=1)
        )
        data = append_signals(data, 1, degree_signals)
        return data


class SimplicialComplexTriangleCoadjacencyDegreeTransform(object):
    def __call__(self, data):
        data = create_signals_on_data_if_needed(data)
        degree_signals = torch.from_numpy(
            data.sc.coadjacency_matrix(2).sum(axis=1)
        )
        data = append_signals(data, 2, degree_signals)
        return data


class OrientableToClassSimplicialComplexTransform(object):
    def __call__(self, data):
        data = create_other_features_on_data_if_needed(data)
        data.other_features["y"] = data.orientable.long()
        return data


class DimOneBoundarySimplicialComplexTransform(object):
    def __call__(self, data):
        data = create_neighborhood_matrices_on_data_if_needed(data)
        data.neighborhood_matrices["1_boundary"] = data.sc.incidence_matrix(1)
        return data


class DimTwoBoundarySimplicialComplexTransform(object):
    def __call__(self, data):
        data = create_neighborhood_matrices_on_data_if_needed(data)
        data.neighborhood_matrices["2_boundary"] = data.sc.incidence_matrix(2)
        return data


class DimZeroHodgeLaplacianSimplicialComplexTransform(object):
    def __call__(self, data):
        data = create_neighborhood_matrices_on_data_if_needed(data)
        data.neighborhood_matrices["0_laplacian"] = (
            data.sc.hodge_laplacian_matrix(rank=0)
        )
        return data


class DimOneHodgeLaplacianUpSimplicialComplexTransform(object):
    def __call__(self, data):
        data = create_neighborhood_matrices_on_data_if_needed(data)
        data.neighborhood_matrices["1_laplacian_up"] = (
            data.sc.up_laplacian_matrix(rank=1)
        )
        return data


class DimOneHodgeLaplacianDownSimplicialComplexTransform(object):
    def __call__(self, data):
        data = create_neighborhood_matrices_on_data_if_needed(data)
        data.neighborhood_matrices["1_laplacian_down"] = (
            data.sc.down_laplacian_matrix(rank=1)
        )
        return data


class DimOneHodgeLaplacianSimplicialComplexTransform(object):
    def __call__(self, data):
        data = create_neighborhood_matrices_on_data_if_needed(data)
        data.neighborhood_matrices["1_laplacian"] = (
            data.sc.hodge_laplacian_matrix(rank=1)
        )
        return data


class DimTwoHodgeLaplacianSimplicialComplexTransform(object):
    def __call__(self, data):
        data = create_neighborhood_matrices_on_data_if_needed(data)
        data.neighborhood_matrices["2_laplacian"] = (
            data.sc.hodge_laplacian_matrix(rank=2)
        )
        return data
