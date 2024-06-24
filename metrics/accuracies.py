import torch
from .metrics import BettiNumbersMetricCollection
from torchmetrics import Metric


def compute_orientability_accuracies(metrics, y_hat, y, name):
    y_hat = torch.sigmoid(y_hat)
    return [{"name": name, "value": metrics(y_hat, y)}]


def compute_name_accuracies(metrics, y_hat, y, name):
    y_hat = torch.sigmoid(y_hat)
    return [{"name": name, "value": metrics(y_hat, y)}]


def compute_betti_numbers_accuracies(
    metrics: BettiNumbersMetricCollection,
    y_hat: torch.Tensor,
    y: torch.Tensor,
    name: torch.Tensor,
):

    metrics_list = metrics.as_list()
    assert y_hat.shape[1] == len(metrics_list)

    res = []
    for dim in range(3):
        # dim is for the type of betti number. e.g. dim=0 refers to betti number $b_0$
        metrics_for_dim = metrics_list[dim]

        for metric_idx in range(len(metrics_list[dim])):
            # different metrics for evaluating each betti number prediction
            metric: Metric = metrics_for_dim[metric_idx].to(y_hat.device)
            res.append(
                {
                    "name": name + f"_betti_{dim}_{metric.__class__.__name__}",
                    "value": metric(
                        torch.max(y_hat[:, dim], torch.tensor(0.0))
                        .round()
                        .long(),
                        y[:, dim].long(),
                    ),
                }
            )
    return res
