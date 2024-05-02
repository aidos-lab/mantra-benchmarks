from torch import nn

from experiments.lightning_modules.BaseModuleOrientability import (
    BaseOrientabilityModule,
)


class GraphCommonModuleOrientability(BaseOrientabilityModule):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x, edge_index, batch):
        x = self.base_model(x, edge_index, batch)
        return x

    def general_step(self, batch, batch_idx, step: str):
        batch_len = len(batch.y)
        x_hat = self(batch.x, batch.edge_index, batch.batch)
        # Squeeze x_hat to match the shape of y
        x_hat = x_hat.squeeze()
        y = batch.y.float()
        loss = nn.functional.binary_cross_entropy_with_logits(x_hat, y)
        self.log(
            f"{step}_loss",
            loss,
            prog_bar=True,
            batch_size=batch_len,
            on_step=False,
            on_epoch=True,
        )
        self.log_accuracies(x_hat, y, batch_len, step)
        return loss
