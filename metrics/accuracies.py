import torch
from .metrics import BettiNumbersMetricCollection, NamedMetric
from torchmetrics import Metric


def compute_orientability_accuracies(
    metrics: NamedMetric, y_hat, y, name: str
):
    y_hat = torch.sigmoid(y_hat)
    metric = metrics.metric.to(y_hat.device)
    return [{"name": f"{name}_{metrics.name}", "value": metric(y_hat, y)}]


def compute_name_accuracies(metrics: NamedMetric, y_hat, y, name: str):
    y_hat = torch.sigmoid(y_hat)
    metric = metrics.metric.to(y_hat.device)
    return [{"name": f"{name}_{metrics.name}", "value": metric(y_hat, y)}]


def compute_betti_numbers_accuracies(
    metrics: BettiNumbersMetricCollection,
    y_hat: torch.Tensor,
    y: torch.Tensor,
    name: str,
):

    metrics_list = metrics.as_list()
    assert y_hat.shape[1] == len(metrics_list)

    res = []
    for dim in range(3):
        # dim is for the type of betti number. e.g. dim=0 refers to betti number $b_0$
        metrics_for_dim = metrics_list[dim]

        for metric_idx in range(len(metrics_list[dim])):
            # different metrics for evaluating each betti number prediction
            metric: Metric = metrics_for_dim[metric_idx].metric.to(
                y_hat.device
            )
            res.append(
                {
                    "name": name
                    + f"_betti_{dim}_{metrics_for_dim[metric_idx].name}",
                    "value": metric(
                        torch.max(y_hat[:, dim], torch.tensor(0.0))
                        .round()
                        .long(),
                        y[:, dim].long(),
                    ),
                }
            )
    return res
