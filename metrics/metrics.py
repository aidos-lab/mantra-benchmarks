import torchmetrics
from torch.nn import ModuleList
import torch
from torch import Tensor
from torchmetrics import Metric
import torchmetrics.classification
from typing import List


class GeneralAccuracy(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state(
            "correct", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self) -> Tensor:
        return self.correct.float() / self.total


def get_orientability_metrics():
    return (
        torchmetrics.classification.F1Score(task="binary"),
        torchmetrics.classification.F1Score(task="binary"),
        torchmetrics.classification.F1Score(task="binary"),
    )


def get_name_metrics(num_classes=5):
    return (
        torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_classes,
            average="weighted",
        ),
        torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_classes, average="weighted"
        ),
        torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_classes, average="weighted"
        ),
    )


def get_betti_numbers_metrics():
    return (
        ModuleList([GeneralAccuracy() for _ in range(3)]),
        ModuleList([GeneralAccuracy() for _ in range(3)]),
        ModuleList([GeneralAccuracy() for _ in range(3)]),
    )
