from typing import Literal

import lightning as L
import torch
import torchmetrics


class BaseClassificationModule(L.LightningModule):
    def __init__(self, task: Literal["orientability", "name"]):
        super().__init__()
        # Accuracy metrics
        if task == "orientability":
            self.training_accuracy = (
                torchmetrics.classification.BinaryAccuracy()
            )
            self.validation_accuracy = (
                torchmetrics.classification.BinaryAccuracy()
            )
            self.test_accuracy = torchmetrics.classification.BinaryAccuracy()
        elif task == "name":
            num_classes = (
                5  # "Klein bottle": 0, "": 1, "RP^2": 2, "T^2": 3, "S^2": 4,
            )
            self.training_accuracy = (
                torchmetrics.classification.MulticlassAccuracy(
                    num_classes=num_classes, average="micro"
                )
            )
            self.validation_accuracy = (
                torchmetrics.classification.MulticlassAccuracy(
                    num_classes=num_classes, average="micro"
                )
            )
            self.test_accuracy = (
                torchmetrics.classification.MulticlassAccuracy(
                    num_classes=num_classes, average="micro"
                )
            )
        else:
            raise ValueError(f"Task {task} not supported")

    def log_accuracies(
        self, x_hat, y, batch_len, step: Literal["train", "test", "validation"]
    ):
        # Apply the sigmoid function to x_hat to get the probabilities
        x_hat = torch.sigmoid(x_hat)
        if step == "train":
            self.training_accuracy(x_hat, y)
            self.log(
                "train_accuracy",
                self.training_accuracy,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch_len,
            )
        elif step == "test":
            self.test_accuracy(x_hat, y)
            self.log(
                "test_accuracy",
                self.test_accuracy,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch_len,
            )
        elif step == "validation":
            self.validation_accuracy(x_hat, y)
            self.log(
                "validation_accuracy",
                self.validation_accuracy,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch_len,
            )

    def test_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "test")

    def validation_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "validation")

    def training_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "train")