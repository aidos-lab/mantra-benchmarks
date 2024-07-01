import torchmetrics
from torch.nn import ModuleList
import torch
from torch import Tensor
from torchmetrics import Metric
import torchmetrics.classification
from typing import Optional, List


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


class BettiNumbersMultiClassAccuracy(Metric):
    def __init__(self, num_classes: int = 7, **kwargs: torch.Any) -> None:
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.acc = torchmetrics.classification.MulticlassAccuracy(
            num_classes=self.num_classes,
            average="macro",
        )

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds = torch.min(
            torch.max(preds, torch.tensor(0.0)),
            torch.tensor(self.num_classes - 1),
        )
        self.acc.update(preds=preds, target=target)

    def compute(self) -> Tensor:
        return self.acc.compute()


class MatthewsCorrCoeff(Metric):
    """
    Binary MCC according to https://en.wikipedia.org/wiki/Phi_coefficient
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.binary_confusion_matrix = (
            torchmetrics.classification.BinaryConfusionMatrix()
        )

    def update(self, preds: Tensor, target: Tensor) -> None:
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")
        preds = torch.min(torch.max(preds, torch.tensor(0.0)), torch.tensor(1))

        self.binary_confusion_matrix.update(preds, target)

    def compute(self) -> Tensor:
        conf_matrix = self.binary_confusion_matrix.compute()

        tn, fp, fn, tp = conf_matrix.flatten()

        denom = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        nom = (tp * tn) - (fp * fn)

        if denom == 0 and nom == 0:
            # model predicts only positives or only negatives. Thus, correlation is 0.
            mcc = 0
        else:
            mcc = nom / denom

        return mcc


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
