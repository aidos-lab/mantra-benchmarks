import torchmetrics
from torch.nn import ModuleList
import torch
from torch import Tensor
from torchmetrics import Metric
import torchmetrics.classification
from typing import Optional, List


class BettiNumbersMetricCollection:
    betti_0: ModuleList
    betti_1: ModuleList
    betti_2: ModuleList

    def __init__(
        self,
        betti_0: List[Metric],
        betti_1: List[Metric],
        betti_2: List[Metric],
    ) -> None:
        self.betti_0 = betti_0
        self.betti_1 = betti_1
        self.betti_2 = betti_2

    def as_list(self):
        return [self.betti_0, self.betti_1, self.betti_2]


class MetricTrainValTest:
    train: Metric | BettiNumbersMetricCollection
    val: Metric | BettiNumbersMetricCollection
    test: Metric | BettiNumbersMetricCollection

    def __init__(
        self,
        train: Metric,
        val: Optional[Metric] = None,
        test: Optional[Metric] = None,
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
        torchmetrics.classification.F1Score(task="binary")
    )
    return metrics


def get_name_metrics(num_classes=5):
    metrics = MetricTrainValTest(
        torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_classes,
            average="weighted",
        )
    )
    return metrics


def get_betti_numbers_metrics():

    betti_0_metrics = ModuleList([GeneralAccuracy()])
    betti_1_metrics = ModuleList([GeneralAccuracy()])
    betti_2_metrics = ModuleList(
        [
            GeneralAccuracy(),
            MatthewsCorrCoeff(),
            torchmetrics.classification.BinaryF1Score(),
        ]
    )

    collection = BettiNumbersMetricCollection(
        betti_0=betti_0_metrics,
        betti_1=betti_1_metrics,
        betti_2=betti_2_metrics,
    )

    metrics = MetricTrainValTest(collection)
    return metrics
