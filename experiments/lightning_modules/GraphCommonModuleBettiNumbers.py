import torch
from torch import nn

from experiments.lightning_modules.BaseModuleBettiNumbers import (
    BaseBettiNumbersModule,
)


class GraphCommonModuleBettiNumbers(BaseBettiNumbersModule):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x, edge_index, batch):
        x = self.base_model(x, edge_index, batch)
        return x

    def general_step(self, batch, batch_idx, step: str):
        x_hat = self(batch.x, batch.edge_index, batch.batch)
        y = torch.tensor(
            batch.betti_numbers, device=x_hat.device, dtype=x_hat.dtype
        )
        batch_len = len(y)
        loss = nn.functional.mse_loss(x_hat, y)
        self.log(
            f"{step}_loss",
            loss,
            prog_bar=True,
            batch_size=batch_len,
            on_step=False,
            on_epoch=True,
        )
        self.log_scores(x_hat, y, batch_len, step)
        return loss
