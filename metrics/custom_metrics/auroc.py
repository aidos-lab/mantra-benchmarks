import torch
from torch import Tensor
from torchmetrics import Metric
import torchmetrics.classification


class AUROC(Metric):
    """
    Multiclass AUROC wrapper for computing AUROC on betti number predictions.
    """

    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.auroc = torchmetrics.classification.AUROC(
            task="multiclass", num_classes=num_classes
        )

    def update(self, preds: Tensor, target: Tensor) -> None:
        y_hat = torch.nn.functional.one_hot(preds, self.num_classes).float()
        self.auroc.update(preds=y_hat, target=target)

    def compute(self) -> Tensor:
        return self.auroc.compute()
