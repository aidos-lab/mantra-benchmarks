from itertools import chain, combinations

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from jaxtyping import Float
from toponetx.classes import SimplicialComplex, CellComplex

from math_utils import eigenvectors_smallest_k_eigenvalues
from models.cells.transformer.positional_encodings.BasePositionalEncodings import (
    BasePositionalEncodings,
)
from models.cells.transformer.positional_encodings.RandomWalks import (
    generate_pe_from_transition_matrix,
)


def barycentric_subdivision_positional_encoding(
    t_complex: SimplicialComplex | CellComplex,
    length_pos_enc: int,
    padding=False,
):
    H, map_cells_graph_vertices = get_barycentric_subdivision_graph_laplacian(
        t_complex, normalize=True
    )
    n = H.shape[0]
    if not padding and n <= length_pos_enc:
        assert (
            "the number of eigenvectors k must be smaller than the number of "
            + f"simplices or cells in the complex n, {length_pos_enc} and {n} detected."
        )
    ev = eigenvectors_smallest_k_eigenvalues(H, length_pos_enc)
    if isinstance(t_complex, CellComplex):
        return _extract_positional_encodings_from_matrix_rows(
            t_complex, ev, map_cells_graph_vertices
        )
    else:
        raise NotImplementedError


def barycentric_subdivision_random_walk_positional_encoding(
    t_complex: SimplicialComplex | CellComplex, length_pos_enc: int
):
    if isinstance(t_complex, CellComplex):
        (
            G,
            map_cells_graph_vertices,
        ) = barycentric_subdivision_1_skeleton_cell_complex(t_complex)
    else:
        (
            G,
            map_cells_graph_vertices,
        ) = barycentric_subdivision_1_skeleton_simplicial_complex(t_complex)
    A = nx.adjacency_matrix(G)
    D = sp.diags(
        np.array(
            [
                0.0 if np.isclose(G.degree[i], 0) else G.degree[i] ** (-1)
                for i in range(G.number_of_nodes())
            ]
        ),
        format="csr",
    )
    RW = A @ D
    random_walk_probs = generate_pe_from_transition_matrix(RW, length_pos_enc)
    if isinstance(t_complex, CellComplex):
        return _extract_positional_encodings_from_matrix_rows(
            t_complex, random_walk_probs, map_cells_graph_vertices
        )
    else:
        raise NotImplementedError


def _extract_positional_encodings_from_matrix_rows(
    cell_complex, m_rows_as_pe, map_cells_graph_vertices
):
    if isinstance(cell_complex, CellComplex):
        nodes = list(cell_complex.nodes)
        edges = list(cell_complex.edges)
        cells = list(cell_complex.cells)
    else:
        raise NotImplementedError
    positional_encodings = []
    vertices_pe = [
        m_rows_as_pe[map_cells_graph_vertices[cell]] for cell in nodes
    ]
    vertices_pe = np.stack(vertices_pe, axis=0)
    positional_encodings.append(vertices_pe)
    if cell_complex.dim >= 1:
        edges_pe = [
            m_rows_as_pe[map_cells_graph_vertices[edge]] for edge in edges
        ]
        edges_pe = np.stack(edges_pe, axis=0)
        positional_encodings.append(edges_pe)
    if cell_complex.dim >= 2:
        faces_pe = [
            m_rows_as_pe[map_cells_graph_vertices[face]] for face in cells
        ]
        faces_pe = np.stack(faces_pe, axis=0)
        positional_encodings.append(faces_pe)
    return positional_encodings


def barycentric_subdivision_1_skeleton_cell_complex(c_complex: CellComplex):
    nodes = list(c_complex.nodes)
    edges = list(c_complex.edges)
    cells = list(c_complex.cells)
    map_cells_graph_vertices = {
        cell: graph_vertex
        for graph_vertex, cell in enumerate(nodes + edges + cells)
    }
    graph = nx.Graph()
    graph.add_nodes_from(range(len(map_cells_graph_vertices)))
    # Add labels to the nodes corresponding to the cells
    nx.set_node_attributes(
        graph,
        {i: cell for i, cell in enumerate(nodes + edges + cells)},
        "cell",
    )
    for edge in edges:
        graph.add_edge(
            map_cells_graph_vertices[edge[0]], map_cells_graph_vertices[edge]
        )
        graph.add_edge(
            map_cells_graph_vertices[edge[1]], map_cells_graph_vertices[edge]
        )
    for cell in cells:
        for vertex in cell.elements:
            graph.add_edge(
                map_cells_graph_vertices[cell],
                map_cells_graph_vertices[vertex],
            )
        for edge in cell.boundary:
            if edge in map_cells_graph_vertices:
                edge_in_graph = map_cells_graph_vertices[edge]
            else:
                edge_in_graph = map_cells_graph_vertices[edge[::-1]]
            graph.add_edge(map_cells_graph_vertices[cell], edge_in_graph)
    return graph, map_cells_graph_vertices


def barycentric_subdivision_1_skeleton_simplicial_complex(
    s_complex: SimplicialComplex,
):
    simplices = [frozenset(s) for s in s_complex.simplices]
    map_cells_graph_vertices = {
        cell: graph_vertex for graph_vertex, cell in enumerate(simplices)
    }
    G = nx.Graph()
    G.add_nodes_from(range(len(map_cells_graph_vertices)))
    for simplex in simplices:
        for face in chain.from_iterable(
            combinations(list(simplex), r) for r in range(1, len(simplex))
        ):
            face_as_set = frozenset(face)
            G.add_edge(
                map_cells_graph_vertices[simplex],
                map_cells_graph_vertices[face_as_set],
            )
    return G, map_cells_graph_vertices


def get_barycentric_subdivision_graph_laplacian(
    t_complex: SimplicialComplex | CellComplex, normalize=False
):
    if isinstance(t_complex, CellComplex):
        (
            G,
            map_cells_graph_vertices,
        ) = barycentric_subdivision_1_skeleton_cell_complex(t_complex)
    else:
        (
            G,
            map_cells_graph_vertices,
        ) = barycentric_subdivision_1_skeleton_simplicial_complex(t_complex)
    if normalize:
        H = nx.normalized_laplacian_matrix(G)
    else:
        H = nx.laplacian_matrix(G)
    return H, map_cells_graph_vertices


class BarycentricSubdivisionRandomWalkPE(BasePositionalEncodings):
    def generate_positional_encodings(
        self,
        t_complex: CellComplex | SimplicialComplex,
        length_positional_encodings: int,
    ) -> dict[int, Float[torch.Tensor, "n_dim length_positional_encodings"]]:
        pes = barycentric_subdivision_random_walk_positional_encoding(
            t_complex=t_complex, length_pos_enc=length_positional_encodings
        )
        pe = {
            i: torch.tensor(pes[i], dtype=torch.float32)
            for i in range(len(pes))
        }
        return pe


class BarycentricSubdivisionEigenvectorsPE(BasePositionalEncodings):
    def generate_positional_encodings(
        self,
        t_complex: CellComplex | SimplicialComplex,
        length_positional_encodings: int,
    ) -> dict[int, Float[torch.Tensor, "n_dim length_positional_encodings"]]:
        pes = barycentric_subdivision_positional_encoding(
            t_complex=t_complex, length_pos_enc=length_positional_encodings
        )
        pe = {
            i: torch.tensor(pes[i], dtype=torch.float32)
            for i in range(len(pes))
        }
        return pe
