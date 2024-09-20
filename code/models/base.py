"""
Base pytorch lightning model.
"""

from typing import Literal
import lightning as L
import torch
import numpy as np


class BaseModel(L.LightningModule):
    def __init__(
        self,
        model,
        training_accuracy,
        test_accuracy,
        validation_accuracy,
        accuracies_fn,
        loss_fn,
        learning_rate,
        imbalance,
    ):
        super().__init__()
        self.training_accuracy = training_accuracy
        self.validation_accuracy = validation_accuracy
        self.test_accuracy = test_accuracy
        self.loss_fn = loss_fn
        self.accuracies_fn = accuracies_fn
        self.model = model
        self.learning_rate = learning_rate
        self.imbalance = imbalance
        if imbalance is not None:
            self.imbalance = np.array(list(self.imbalance))
            self.imbalance = self.imbalance / np.sum(self.imbalance)

    def forward(self, batch):
        x = self.model(batch)
        return x

    def get_log_name(self, log_type: str, step: str):
        loss_log_name = f"{step}_{log_type}"
        return loss_log_name

    def general_step(self, batch, batch_idx, step: str):

        # This is rather ugly, open to better solutions,
        # but torch_geometric and the toponetx dl have rather different
        # signatures.
        if hasattr(batch, "batch_size"):
            batch_len = batch.batch_size
        elif hasattr(batch, "batch"):
            batch_len = batch.batch.max() + 1
        else:
            raise ValueError(
                "Batch object does not have a known way to compute batch size."
            )

        # Generalizing to accomodate for the different signatures.
        x_hat = self(batch)
        # Squeeze x_hat to match the shape of y
        x_hat = x_hat.squeeze()
        imbalance = (
            torch.tensor(
                self.imbalance, device=self.device, dtype=torch.float32
            )
            if self.imbalance is not None
            else None
        )
        loss = self.loss_fn(
            x_hat,
            batch.y,
            weight=imbalance,
        )

        self.log(
            self.get_log_name("loss", step),
            loss,
            prog_bar=True,
            batch_size=batch_len,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log_accuracies(x_hat, batch.y, batch_len, step)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        return optimizer

    def log_accuracies(
        self, x_hat, y, batch_len, step: Literal["train", "test", "validation"]
    ):
        if step == "train":
            acc_fun = self.training_accuracy
        elif step == "test":
            acc_fun = self.test_accuracy
        elif step == "validation":
            acc_fun = self.validation_accuracy
        else:
            ValueError("Unknown step Literal")

        accuracies = self.accuracies_fn(acc_fun, x_hat, y, step)
        for accuracy in accuracies:
            self.log(
                **accuracy,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch_len,
                sync_dist=True,
            )

    def test_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "test")

    def validation_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "validation")

    def training_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "train")
