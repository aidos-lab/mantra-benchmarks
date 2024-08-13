import torch.nn as nn
import torch


def name_loss_fn(y_pred, y, weight: torch.Tensor):
    return nn.functional.cross_entropy(y_pred, y, weight)


def betti_loss_fn(y_pred, y, weight=None):
    return nn.functional.mse_loss(y_pred, y)


def orientability_loss_fn(y_pred, y, weight=torch.Tensor):
    weight_tensor = y.float() * weight[0] + (1 - y.float()) * weight[1]
    return nn.functional.binary_cross_entropy_with_logits(
        y_pred, y.float(), weight=weight_tensor
    )
