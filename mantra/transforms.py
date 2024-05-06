import gudhi
import numpy as np
import torch
from toponetx.classes import SimplicialComplex
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import degree

import k_simplex2vec as ks2v
from mantra.utils import (
    create_signals_on_data_if_needed,
    append_signals,
    create_other_features_on_data_if_needed,
    create_neighborhood_matrices_on_data_if_needed,
)


class SetNumNodesTransform:
    def __call__(self, data):
        data.num_nodes = data.n_vertices
        return data


class Simplex2VecTransform:
    def __call__(self, data):
        st = gudhi.SimplexTree()

        ei = [
            [edge[0], edge[1]]
            for edge in data.edge_index.T.tolist()
            if edge[0] < edge[1]
        ]
        data.edge_index = torch.tensor(ei).T
        # Say hi to bad programming
        for edge in ei:
            st.insert(edge)
        st.expansion(3)

        p1 = ks2v.assemble(cplx=st, k=1, scheme="uniform", laziness=None)
        P1 = p1.toarray()

        Simplices = list()
        for simplex in st.get_filtration():
            if simplex[1] != np.inf:
                Simplices.append(simplex[0])
            else:
                break

        ## Perform random walks on the edges
        L = 20
        N = 40
        Walks = ks2v.RandomWalks(walk_length=L, number_walks=N, P=P1, seed=3)
        # to save the walks in a text file
        ks2v.save_random_walks(Walks, "RandomWalks_Edges.txt")

        ## Embed the edges
        Emb = ks2v.Embedding(
            Walks=Walks,
            emb_dim=20,
            epochs=5,
            filename="k-simplex2vec_Edge_embedding.model",
        )
        data.edge_attr = torch.tensor(Emb.wv.vectors)
        toundirected = ToUndirected()
        data = toundirected(data)
        return data


class OrientableToClassTransform:
    def __call__(self, data):
        data.y = data.orientable.long()
        return data


class DegreeTransform:
    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        data.x = deg.view(-1, 1)
        return data


class TriangulationToFaceTransform:
    def __call__(self, data):
        data.face = torch.tensor(data.triangulation).T - 1
        data.triangulation = None
        return data


class SimplicialComplexTransform:
    def __call__(self, data):
        data.sc = SimplicialComplex(data.triangulation)
        create_signals_on_data_if_needed(data)
        create_other_features_on_data_if_needed(data)
        create_neighborhood_matrices_on_data_if_needed(data)
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
        data = create_other_features_on_data_if_needed(data)
        data.other_features["y"] = data.orientable.long()
        return data


class DimOneBoundarySimplicialComplexTransform:
    def __call__(self, data):
        data = create_neighborhood_matrices_on_data_if_needed(data)
        data.neighborhood_matrices["1_boundary"] = data.sc.incidence_matrix(1)
        return data


class DimTwoBoundarySimplicialComplexTransform:
    def __call__(self, data):
        data = create_neighborhood_matrices_on_data_if_needed(data)
        data.neighborhood_matrices["2_boundary"] = data.sc.incidence_matrix(2)
        return data


class DimZeroHodgeLaplacianSimplicialComplexTransform:
    def __call__(self, data):
        data = create_neighborhood_matrices_on_data_if_needed(data)
        data.neighborhood_matrices["0_laplacian"] = (
            data.sc.hodge_laplacian_matrix(rank=0)
        )
        return data


class DimOneHodgeLaplacianUpSimplicialComplexTransform:
    def __call__(self, data):
        data = create_neighborhood_matrices_on_data_if_needed(data)
        data.neighborhood_matrices["1_laplacian_up"] = (
            data.sc.up_laplacian_matrix(rank=1)
        )
        return data


class DimOneHodgeLaplacianDownSimplicialComplexTransform:
    def __call__(self, data):
        data = create_neighborhood_matrices_on_data_if_needed(data)
        data.neighborhood_matrices["1_laplacian_down"] = (
            data.sc.down_laplacian_matrix(rank=1)
        )
        return data


class DimOneHodgeLaplacianSimplicialComplexTransform:
    def __call__(self, data):
        data = create_neighborhood_matrices_on_data_if_needed(data)
        data.neighborhood_matrices["1_laplacian"] = (
            data.sc.hodge_laplacian_matrix(rank=1)
        )
        return data


class DimTwoHodgeLaplacianSimplicialComplexTransform:
    def __call__(self, data):
        data = create_neighborhood_matrices_on_data_if_needed(data)
        data.neighborhood_matrices["2_laplacian"] = (
            data.sc.hodge_laplacian_matrix(rank=2)
        )
        return data


class NameToClassTransform:
    def __init__(self):
        self.class_dict = {
            "Klein bottle": 0,
            "": 1,
            "RP^2": 2,
            "T^2": 3,
            "S^2": 4,
        }

    def __call__(self, data):
        # data.y = F.one_hot(torch.tensor(self.class_dict[data.name]),num_classes=5)
        data.y = torch.tensor(self.class_dict[data.name])
        return data


class NameToClassSimplicialComplexTransform:
    def __init__(self):
        self.class_dict = {
            "Klein bottle": 0,
            "": 1,
            "RP^2": 2,
            "T^2": 3,
            "S^2": 4,
        }

    def __call__(self, data):
        data = create_other_features_on_data_if_needed(data)
        data.other_features["y"] = torch.tensor([self.class_dict[data.name]])
        return data
