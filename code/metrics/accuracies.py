import torch
from .metrics import BettiNumbersMetricCollection, NamedMetric
from torchmetrics import Metric
from typing import List


def compute_orientability_accuracies(
    metrics: List[NamedMetric], y_hat, y, name: str
):
    benchmarks = []
    for metrics_ in metrics:
        y_hat_ = torch.sigmoid(y_hat).long()
        metric = metrics_.metric.to(y_hat.device)
        benchmarks.append(
            {"name": f"{name}_{metrics_.name}", "value": metric(y_hat_, y)}
        )
    return benchmarks


def compute_name_accuracies(metrics: List[NamedMetric], y_hat, y, name: str):
    benchmarks = []
    for metrics_ in metrics:
        y_hat = torch.sigmoid(y_hat)
        metric = metrics_.metric.to(y_hat.device)
        benchmarks.append(
            {
                "name": f"{name}_{metrics_.name}",
                "value": metric(y_hat, y),
            }
        )
    return benchmarks


def compute_betti_numbers_accuracies(
    metrics: BettiNumbersMetricCollection,
    y_hat: torch.Tensor,
    y: torch.Tensor,
    name: str,
):

    metrics_list = metrics.as_list()

    assert y_hat.shape[1] == len(metrics_list)

    res = []
    for dim in range(len(metrics_list)):
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
