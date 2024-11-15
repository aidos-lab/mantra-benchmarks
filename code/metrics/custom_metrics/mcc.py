import torch
import torchmetrics.classification
from torch import Tensor
from torchmetrics import Metric


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
