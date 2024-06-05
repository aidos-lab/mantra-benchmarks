import torchmetrics
from torch.nn import ModuleList
from experiments.metrics import GeneralAccuracy


def get_orientability_metrics():
    return (
        torchmetrics.classification.BinaryAccuracy(),
        torchmetrics.classification.BinaryAccuracy(),
        torchmetrics.classification.BinaryAccuracy(),
    )


def get_name_metrics(num_classes=5):
    return (
        torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_classes, average="micro"
        ),
        torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_classes, average="micro"
        ),
        torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_classes, average="micro"
        ),
    )


def get_betti_numbers_metrics():
    return (
        ModuleList([GeneralAccuracy() for _ in range(3)]),
        ModuleList([GeneralAccuracy() for _ in range(3)]),
        ModuleList([GeneralAccuracy() for _ in range(3)]),
    )
