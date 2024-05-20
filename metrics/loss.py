import torch.nn as nn


def name_loss_fn(y_pred, y):
    return nn.functional.cross_entropy(y_pred, y)


def betti_loss_fn(y_pred, y):
    return nn.functional.mse_loss(y_pred, y)


def orientability_loss_fn(y_pred, y):
    return nn.functional.binary_cross_entropy_with_logits(y_pred, y.float())
