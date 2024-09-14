import numpy as np
import torch


def create_signals_on_data_if_needed(data):
    if not hasattr(data, "x") or data.x is None:
        data.x = {}
    return data


def create_or_empty_signals_on_data(data):
    data.x = {}
    return data


def get_triangles_from_simplicial_complex(data):
    try:
        sc = data.sc
    except AttributeError:
        raise AttributeError(
            "Simplicial complex not found in data. Did you apply the SimplicialComplex transform"
            "before trying to get triangulations?"
        )
    triangles = []
    for triangle in sc.skeleton(2):
        triangles.append(list(triangle))
    return triangles


def generate_zero_sparse_connectivity(m, n):
    # Function extracted from TopoBenchmarkX
    """Generate a zero sparse connectivity matrix.

    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns.

    Returns
    -------
    torch.sparse_coo_tensor
        Zero sparse connectivity matrix.
    """
    return torch.sparse_coo_tensor((m, n)).coalesce()


def get_complex_connectivity(complex, max_rank, signed=False):
    # Function extracted from TopoBenchmarkX
    """Get the connectivity matrices for the complex.

    Parameters
    ----------
    complex : topnetx.CellComplex or topnetx.SimplicialComplex
        Cell complex.
    max_rank : int
        Maximum rank of the complex.
    signed : bool, optional
        If True, returns signed connectivity matrices.

    Returns
    -------
    dict
        Dictionary containing the connectivity matrices.
    """
    practical_shape = list(
        np.pad(list(complex.shape), (0, max_rank + 1 - len(complex.shape)))
    )
    connectivity = {}
    for rank_idx in range(max_rank + 1):
        for connectivity_info in [
            "incidence",
            "down_laplacian",
            "up_laplacian",
            "adjacency",
            "hodge_laplacian",
        ]:
            try:
                connectivity[f"{connectivity_info}_{rank_idx}"] = from_sparse(
                    getattr(complex, f"{connectivity_info}_matrix")(
                        rank=rank_idx, signed=signed
                    )
                )
            except ValueError:
                if connectivity_info == "incidence":
                    connectivity[f"{connectivity_info}_{rank_idx}"] = (
                        generate_zero_sparse_connectivity(
                            m=practical_shape[rank_idx - 1],
                            n=practical_shape[rank_idx],
                        )
                    )
                else:
                    connectivity[f"{connectivity_info}_{rank_idx}"] = (
                        generate_zero_sparse_connectivity(
                            m=practical_shape[rank_idx],
                            n=practical_shape[rank_idx],
                        )
                    )
    """
    Not needed right now according to TopoBenchmarkX
    # Obtain normalized incidence matrices
    B1N, B1TN, B2N, B2TN = compute_bunch_normalized_matrices(connectivity['incidence_1'], connectivity['incidence_2'])
    connectivity['incidence_1_normalized'] = B1N
    connectivity['incidence_1_transposed_normalized'] = B1TN
    connectivity['incidence_2_normalized'] = B2N
    connectivity['incidence_2_transposed_normalized'] = B2TN
    connectivity['up_laplacian_0_normalized'] = from_sparse(compute_x_laplacian_normalized_matrix
                                                            (connectivity['hodge_laplacian_0'],
                                                             connectivity['up_laplacian_0']))
    connectivity['up_laplacian_1_normalized'] = from_sparse(compute_x_laplacian_normalized_matrix
                                                            (connectivity['hodge_laplacian_1'],
                                                             connectivity['up_laplacian_1']))
    connectivity['down_laplacian_1_normalized'] = from_sparse(compute_x_laplacian_normalized_matrix
                                                              (connectivity['hodge_laplacian_1'],
                                                               connectivity['down_laplacian_1']))
    connectivity['down_laplacian_2_normalized'] = from_sparse(compute_x_laplacian_normalized_matrix
                                                              (connectivity['hodge_laplacian_2'],
                                                               connectivity['down_laplacian_2']))
                                                               
   connectivity["shape"] = practical_shape
    """

    return connectivity


def append_signals(data, signals_key, signals):
    if signals_key not in data.x:
        data.x[signals_key] = signals
    else:
        data.x[signals_key] = torch.concatenate([data.x[1], signals], dim=1)
    return data


def from_sparse(data, device=None) -> torch.Tensor:
    """Convert sparse input data directly to torch sparse coo format.

    Parameters
    ----------
    device: Torch device where we want to store the data.
    data : scipy.sparse._csc.csc_matrix
        Input n_dimensional data.

    Returns
    -------
    torch.sparse_coo, same shape as data
        input data converted to tensor.
    """
    if device is None:
        device = torch.device("cpu")
    # cast from csc_matrix to coo format for compatibility
    coo = data.tocoo()

    values = torch.FloatTensor(coo.data)
    values = values.to(device)
    indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
    indices = indices.to(device)
    sparse_data = torch.sparse_coo_tensor(
        indices, values, coo.shape
    ).coalesce()
    sparse_data = sparse_data.to(device)
    return sparse_data


def transfer_simplicial_complex_batch_to_device(batch, device, dataloader_idx):
    batched_example, signals_belongings, len_batch = batch
    for key in batched_example.signals:
        batched_example.signals[key] = batched_example.signals[key].to(device)
    for key in batched_example.other_features:
        batched_example.other_features[key] = batched_example.other_features[
            key
        ].to(device)
    if batched_example.neighborhood_matrices is not None:
        for key in batched_example.neighborhood_matrices:
            batched_example.neighborhood_matrices[key] = from_sparse(
                batched_example.neighborhood_matrices[key], device=device
            )
    for key in signals_belongings:
        signals_belongings[key] = signals_belongings[key].to(device)
    # Not using batch masks
    return batched_example, signals_belongings, len_batch
