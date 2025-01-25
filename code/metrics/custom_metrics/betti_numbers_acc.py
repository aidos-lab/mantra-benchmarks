import torch
import torchmetrics.classification
from torch import Tensor
from torchmetrics import Metric


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
