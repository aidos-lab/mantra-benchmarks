from typing import Literal

import lightning as L
from torch.nn import ModuleList

from experiments.metrics import GeneralAccuracy


class BaseBettiNumbersModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        # 3 accuracies per modality: one for each betti number
        self.training_accuracies = ModuleList(
            [GeneralAccuracy() for _ in range(3)]
        )
        self.validation_accuracies = ModuleList(
            [GeneralAccuracy() for _ in range(3)]
        )
        self.test_accuracies = ModuleList(
            [GeneralAccuracy() for _ in range(3)]
        )

    def log_scores(
        self, x_hat, y, batch_len, step: Literal["train", "test", "validation"]
    ):
        # x_hat is a float tensor of shape (batch_len, 3), one column per betti number and row per sample
        # y is a long tensor of shape (batch_len, 3), one row per sample and column per betti number
        if step == "train":
            for dim in range(3):
                self.training_accuracies[dim](
                    x_hat[:, dim].round().long(), y[:, dim].long()
                )
                self.log(
                    f"training_accuracy_betti_{dim}",
                    self.training_accuracies[dim],
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_len,
                )

        elif step == "test":
            for dim in range(3):
                self.test_accuracies[dim](
                    x_hat[:, dim].round().long(), y[:, dim].long()
                )
                self.log(
                    f"test_accuracy_betti_{dim}",
                    self.test_accuracies[dim],
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_len,
                )
        elif step == "validation":
            for dim in range(3):
                self.validation_accuracies[dim](
                    x_hat[:, dim].round().long(), y[:, dim].long()
                )
                self.log(
                    f"validation_accuracy_betti_{dim}",
                    self.validation_accuracies[dim],
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
