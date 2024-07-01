import torchmetrics
import torch
from torch import Tensor
from torchmetrics import Metric
import torchmetrics.classification
from typing import Optional, List
from metrics.custom_metrics.general_accuracy import GeneralAccuracy
from metrics.custom_metrics.betti_numbers_acc import (
    BettiNumbersMultiClassAccuracy,
)
from metrics.custom_metrics.mcc import MatthewsCorrCoeff


class NamedMetric:
    metric: Metric
    name: str

    def __init__(self, metric: Metric, name: str) -> None:
        self.metric = metric
        self.name = name


class BettiNumbersMetricCollection:
    betti_0: List[NamedMetric]
    betti_1: List[NamedMetric]
    betti_2: List[NamedMetric]

    def __init__(
        self,
        betti_0: List[NamedMetric],
        betti_1: List[NamedMetric],
        betti_2: List[NamedMetric],
    ) -> None:
        self.betti_0 = betti_0
        self.betti_1 = betti_1
        self.betti_2 = betti_2

    def as_list(self):
        return [self.betti_0, self.betti_1, self.betti_2]


class MetricTrainValTest:
    train: List[NamedMetric] | BettiNumbersMetricCollection
    val: List[NamedMetric] | BettiNumbersMetricCollection
    test: List[NamedMetric] | BettiNumbersMetricCollection

    def __init__(
        self,
        train: List[NamedMetric] | BettiNumbersMetricCollection,
        val: Optional[List[NamedMetric]] = None,
        test: Optional[List[NamedMetric]] = None,
    ) -> None:
        self.train = train
        self.val = self.train if val is None else val
        self.test = self.train if test is None else test


def get_orientability_metrics():
    metrics = MetricTrainValTest(
        [
            NamedMetric(
                torchmetrics.classification.F1Score(task="binary"), "F1Score"
            ),
            NamedMetric(MatthewsCorrCoeff(), "MCC"),
        ]
    )
    return metrics


def get_name_metrics(num_classes=5):
    metrics = MetricTrainValTest(
        [
            NamedMetric(
                torchmetrics.classification.MulticlassAccuracy(
                    num_classes=num_classes,
                    average="weighted",
                ),
                "Accuracy",
            ),
            NamedMetric(
                torchmetrics.classification.MulticlassAccuracy(
                    num_classes=num_classes,
                    average="macro",
                ),
                "BalancedAccuracy",
            ),
        ]
    )
    return metrics


def get_betti_numbers_metrics():

    accuracy_only = [NamedMetric(GeneralAccuracy(), "Accuracy")]

    betti_0_metrics = [NamedMetric(GeneralAccuracy(), "Accuracy")]
    betti_1_metrics = [
        NamedMetric(GeneralAccuracy(), "Accuracy"),
        NamedMetric(
            BettiNumbersMultiClassAccuracy(num_classes=7),
            "BalancedAccuracy",
        ),
    ]

    betti_2_metrics = [
        NamedMetric(GeneralAccuracy(), "Accuracy"),
        NamedMetric(MatthewsCorrCoeff(), "MCC"),
        # NamedMetric(torchmetrics.classification.BinaryF1Score(), "F1"),
        NamedMetric(
            BettiNumbersMultiClassAccuracy(num_classes=2),
            "BalancedAccuracy",
        ),
    ]

    collection = BettiNumbersMetricCollection(
        betti_0=betti_0_metrics,
        betti_1=betti_1_metrics,
        betti_2=betti_2_metrics,
    )

    collection_train = BettiNumbersMetricCollection(
        betti_0=accuracy_only, betti_1=accuracy_only, betti_2=accuracy_only
    )

    metrics = MetricTrainValTest(
        train=collection_train, val=collection_train, test=collection
    )
    return metrics
