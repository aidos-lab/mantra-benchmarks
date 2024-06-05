from typing import Literal
import lightning as L
import torch


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
    ):
        super().__init__()
        self.training_accuracy = training_accuracy
        self.validation_accuracy = validation_accuracy
        self.test_accuracy = test_accuracy
        self.loss_fn = loss_fn
        self.accuracies_fn = accuracies_fn
        self.model = model
        self.learning_rate = learning_rate

    def forward(self, x, edge_index, batch):
        x = self.model(x, edge_index, batch)
        return x

    def general_step(self, batch, batch_idx, step: str):
        batch_len = len(batch.y)
        x_hat = self(batch.x, batch.edge_index, batch.batch)
        # Squeeze x_hat to match the shape of y
        x_hat = x_hat.squeeze()
        loss = self.loss_fn(x_hat, batch.y)
        self.log(
            f"{step}_loss",
            loss,
            prog_bar=True,
            batch_size=batch_len,
            on_step=False,
            on_epoch=True,
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
            accuracies = self.accuracies_fn(
                self.training_accuracy,
                x_hat,
                y,
                "train_accuracy",
            )
            for accuracy in accuracies:
                self.log(
                    **accuracy,
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_len,
                )
        elif step == "test":
            accuracies = self.accuracies_fn(
                self.test_accuracy, x_hat, y, "test_accuracy"
            )
            for accuracy in accuracies:
                self.log(
                    **accuracy,
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_len,
                )
        elif step == "validation":
            accuracies = self.accuracies_fn(
                self.validation_accuracy,
                x_hat,
                y,
                "validation_accuracy",
            )
            for accuracy in accuracies:
                self.log(
                    **accuracy,
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
