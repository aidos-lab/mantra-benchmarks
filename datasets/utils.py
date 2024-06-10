import numpy as np
import torch


def create_signals_on_data_if_needed(data):
    if not hasattr(data, "x") or data.x is None:
        data.x = {}
    return data


def create_other_features_on_data_if_needed(data):
    if not hasattr(data, "other_features") or data.other_features is None:
        data.other_features = {}
    return data


def create_neighborhood_matrices_on_data_if_needed(data):
    if (
        not hasattr(data, "neighborhood_matrices")
        or data.neighborhood_matrices is None
    ):
        data.neighborhood_matrices = {}
    return data


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
