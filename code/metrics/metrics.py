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
from metrics.custom_metrics.auroc import AUROC
from datasets.dataset_types import DatasetType


class NamedMetric:
    """
    torchmetrics.Metric with a name. Name is later used for annotating the result in the benchmark .csv file.
    """

    metric: Metric
    name: str

    def __init__(self, metric: Metric, name: str) -> None:
        self.metric = metric
        self.name = name


class BettiNumbersMetricCollection:
    """
    Class containing the metrics for benchmarking performance on betti number prediction.

    Different metrics for the different betti number types can be specified.
    """

    betti_0: List[NamedMetric]
    betti_1: List[NamedMetric]
    betti_2: List[NamedMetric]
    betti_3: Optional[List[NamedMetric]]

    def __init__(
        self,
        betti_0: List[NamedMetric],
        betti_1: List[NamedMetric],
        betti_2: List[NamedMetric],
        betti_3: Optional[List[NamedMetric]] = None,
    ) -> None:
        self.betti_0 = betti_0
        self.betti_1 = betti_1
        self.betti_2 = betti_2
        self.betti_3 = betti_3

    def as_list(self):
        if self.betti_3 is None:
            return [self.betti_0, self.betti_1, self.betti_2]
        else:
            return [self.betti_0, self.betti_1, self.betti_2, self.betti_3]


class MetricTrainValTest:
    """
    Wrapper class for the metrics during training, validation and testing phase. Allows for using different metrics during testing and training.
    """

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


def get_orientability_metrics(ds_type: DatasetType):
    metrics = MetricTrainValTest(
        [
            NamedMetric(
                torchmetrics.classification.BinaryAUROC(), "Binary_AUROC"
            ),
            NamedMetric(
                torchmetrics.classification.MulticlassAccuracy(
                    num_classes=2,
                    average="macro",
                ),
                "BalancedAccuracy",
            ),
            NamedMetric(GeneralAccuracy(), "Accuracy"),
            NamedMetric(MatthewsCorrCoeff(), "MCC"),
        ]
    )
    return metrics


def get_name_metrics(ds_type: DatasetType):
    if ds_type == DatasetType.FULL_2D:
        num_classes = 5
    elif ds_type == DatasetType.NO_NAMELESS_2D:
        num_classes = 4
    elif ds_type == DatasetType.FULL_3D:
        raise ValueError("name task not allowed on 3 manifolds")
    else:
        raise ValueError("Unknown dataset type")

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
                torchmetrics.classification.AUROC(
                    num_classes=num_classes, task="multiclass"
                ),
                "AUROC",
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


def get_betti_numbers_metrics(ds_type: DatasetType):
    accuracy_only = [NamedMetric(GeneralAccuracy(), "Accuracy")]
    if ds_type == DatasetType.FULL_2D or ds_type == DatasetType.NO_NAMELESS_2D:
        betti_0_metrics = [NamedMetric(GeneralAccuracy(), "Accuracy")]
        betti_1_metrics = [
            NamedMetric(GeneralAccuracy(), "Accuracy"),
            NamedMetric(AUROC(num_classes=7), "AUROC"),
            NamedMetric(
                BettiNumbersMultiClassAccuracy(num_classes=7),
                "BalancedAccuracy",
            ),
        ]

        betti_2_metrics = [
            NamedMetric(GeneralAccuracy(), "Accuracy"),
            NamedMetric(MatthewsCorrCoeff(), "MCC"),
            NamedMetric(
                torchmetrics.classification.BinaryAUROC(), "BinaryAUROC"
            ),
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
            train=collection_train, val=collection, test=collection
        )
    else:
        collection = BettiNumbersMetricCollection(
            betti_0=accuracy_only,
            betti_1=accuracy_only,
            betti_2=accuracy_only,
            betti_3=accuracy_only,
        )
        metrics = MetricTrainValTest(
            train=collection, val=collection, test=collection
        )

    return metrics
