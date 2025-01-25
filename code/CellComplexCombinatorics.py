import scipy
from toponetx import CellComplex
from toponetx.utils import incidence_to_adjacency


def cc_incidence_matrix(
    cell_complex: CellComplex,
    rank: int,
    signed: bool = True,
    index: bool = False,
) -> scipy.sparse.csr_matrix | tuple[dict, dict, scipy.sparse.csr_matrix]:
    """
    Same function as toponet but returns the boundary matrix of the cell complex in the correct order.
    """
    nodelist = (
        cell_complex.nodes
    )  # order simplices as they appear in the cell complex
    if rank == 0:
        A = scipy.sparse.lil_matrix((0, len(nodelist)))
        if index:
            node_index = {node: i for i, node in enumerate(nodelist)}
            if signed:
                return {}, node_index, A.asformat("csr")
            return {}, node_index, abs(A.asformat("csr"))

        if signed:
            return A.asformat("csr")
        return abs(A.asformat("csr"))
    # For rank 1 and 2, we need to have the indices of the vertices.
    node_index = {node: i for i, node in enumerate(nodelist)}
    # edgelist contains edges composed by the indices of the vertices they contain sorted,
    # this is, with the induced orientation by the order of the vertices in the cell complex.
    edgelist = [
        sorted((node_index[e[0]], node_index[e[1]]))
        for e in cell_complex.edges
    ]
    if rank == 1:
        A = scipy.sparse.lil_matrix((len(nodelist), len(edgelist)))
        for ei, e in enumerate(edgelist):
            (ui, vi) = e[
                :2
            ]  # Note that the indices are sorted, so we are orienting the edges
            # by the order of the nodes in the cell complex given by their indices
            A[ui, ei] = -1
            A[vi, ei] = 1
        if index:
            edge_index = {edge: i for i, edge in enumerate(edgelist)}
            if signed:
                return node_index, edge_index, A.asformat("csr")
            return node_index, edge_index, abs(A.asformat("csr"))
        if signed:
            return A.asformat("csr")
        return abs(A.asformat("csr"))
    if rank == 2:
        A = scipy.sparse.lil_matrix((len(edgelist), len(cell_complex.cells)))
        edge_index = {
            tuple(edge): i for i, edge in enumerate(edgelist)
        }  # oriented edges indices
        for celli, cell in enumerate(cell_complex.cells):
            edge_visiting_dic = {}  # this dictionary is cell dependent
            # mainly used to handle the cell complex non-regular case
            for edge in cell.boundary:
                edge_w_indices = (node_index[edge[0]], node_index[edge[1]])
                ei = edge_index[tuple(sorted(edge_w_indices))]
                if ei not in edge_visiting_dic:
                    if edge_w_indices in edge_index:
                        edge_visiting_dic[ei] = 1
                    else:
                        edge_visiting_dic[ei] = -1
                else:
                    if edge in edge_index:
                        edge_visiting_dic[ei] = edge_visiting_dic[ei] + 1
                    else:
                        edge_visiting_dic[ei] = edge_visiting_dic[ei] - 1

                A[ei, celli] = edge_visiting_dic[
                    ei
                ]  # this will update everytime we visit this edge for non-regular cell complexes
                # the regular case can be handled more efficiently :
                # if edge in edge_index:
                #    A[ei, celli] = 1s
                # else:
                #    A[ei, celli] = -1
        if index:
            cell_index = {
                c.elements: i for i, c in enumerate(cell_complex.cells)
            }
            if signed:
                return edge_index, cell_index, A.asformat("csr")
            return edge_index, cell_index, abs(A.asformat("csr"))

        if signed:
            return A.asformat("csr")
        return abs(A.asformat("csr"))
    raise ValueError(f"Only dimensions 0, 1 and 2 are supported, got {rank}.")


def hodge_laplacian_matrix(
    cell_complex: CellComplex,
    rank: int,
    signed: bool = True,
) -> scipy.sparse.csr_matrix:
    assert (
        cell_complex.dim >= rank >= 0
    )  # No negative dimensional Hodge Laplacian
    if cell_complex.dim > rank >= 0:
        up_laplacian = up_laplacian_matrix(cell_complex, rank, True)
    else:
        up_laplacian = None
    if cell_complex.dim >= rank > 0:
        down_laplacian = down_laplacian_matrix(cell_complex, rank, True)
    else:
        down_laplacian = None
    if up_laplacian is not None and down_laplacian is not None:
        hodge_laplacian = up_laplacian + down_laplacian
    elif up_laplacian is not None:
        hodge_laplacian = up_laplacian
    elif down_laplacian is not None:
        hodge_laplacian = down_laplacian
    else:
        # Dimension is 0 because, if the dimension of the cell complex is one or higher,
        # we have at least lower laplacians. Also, if the dimension is greater than zero,
        # we have at least upper laplacians for all dimensions except for cell_complex.dim, for
        # which we have lower laplacian.
        hodge_laplacian = scipy.sparse.coo_matrix(
            (len(cell_complex.nodes), len(cell_complex.nodes))
        )
    if not signed:
        hodge_laplacian = abs(hodge_laplacian)
    return hodge_laplacian


def up_laplacian_matrix(
    cell_complex: CellComplex,
    rank: int,
    signed: bool = True,
) -> scipy.sparse.csr_matrix:
    """
    Same function as toponet but returns the upper laplacian of the cell complex in the correct order.
    """

    if cell_complex.dim > rank >= 0:
        B_next = cc_incidence_matrix(cell_complex, rank + 1)
        L_up = B_next @ B_next.transpose()
    else:
        raise ValueError(
            f"Rank should be larger or equal than 0 and <= {cell_complex.dim - 1} (maximal dimension cells-1), got {rank}"
        )
    if not signed:
        L_up = abs(L_up)
    return L_up


def down_laplacian_matrix(
    cell_complex: CellComplex, rank: int, signed: bool = True, weight=None
) -> scipy.sparse.csr_matrix:
    """
    Same function as toponet but returns the lower laplacian of the cell complex in the correct order.
    """
    if weight is not None:
        raise ValueError("`weight` is not supported in this version")

    if cell_complex.dim >= rank > 0:
        B = cc_incidence_matrix(cell_complex, rank)
        L_down = B.transpose() @ B
    else:
        raise ValueError(
            f"Rank should be larger or equal than 1 and <= {cell_complex.dim} (maximal dimension cells), got {rank}."
        )
    if not signed:
        L_down = abs(L_down)
    return L_down


def lower_adjacency(
    cell_complex: CellComplex, dim: int, s: int = 1
) -> scipy.sparse.spmatrix:
    # A cell is neighbor of itself and all the other cells appearing in the lower hodge laplacian.
    if dim == 0:
        return scipy.sparse.coo_matrix(
            (len(cell_complex.nodes), len(cell_complex.nodes))
        )
    else:
        B = cc_incidence_matrix(cell_complex, dim, signed=False)
        A = incidence_to_adjacency(B, s=s)
        return A.tocoo()


def upper_adjacency(
    cell_complex: CellComplex, dim: int, s: int = 1
) -> scipy.sparse.spmatrix:
    if cell_complex.dim == dim:
        match dim:
            case 0:
                return scipy.sparse.coo_matrix(
                    (len(cell_complex.nodes), len(cell_complex.nodes))
                )
            case 1:
                return scipy.sparse.coo_matrix(
                    (len(cell_complex.edges), len(cell_complex.edges))
                )
            case 2:
                return scipy.sparse.coo_matrix(
                    (len(cell_complex.cells), len(cell_complex.cells))
                )
    else:
        # A cell is neighbor of itself and all the other cells appearing in the upper hodge laplacian.
        B_T = cc_incidence_matrix(
            cell_complex, dim + 1, signed=False
        ).transpose()
        A = incidence_to_adjacency(B_T, s=s)
        return A.tocoo()
