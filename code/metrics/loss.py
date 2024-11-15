import torch
import torch.nn as nn


def name_loss_fn(y_pred, y, weight: torch.Tensor | None):
    return nn.functional.cross_entropy(y_pred, y, weight)


def betti_loss_fn(y_pred, y, weight=None):
    # ignore weighting
    return nn.functional.mse_loss(y_pred, y)


def orientability_loss_fn(y_pred, y, weight: torch.Tensor | None):
    weight_tensor = None
    if weight is not None:
        weight_tensor = y.float() * weight[0] + (1 - y.float()) * weight[1]
    return nn.functional.binary_cross_entropy_with_logits(
        y_pred, y.float(), weight=weight_tensor
    )
