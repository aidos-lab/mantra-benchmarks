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
