from typing import Literal

import torch
from topomodelx.nn.simplicial.scnn import SCNN
from torch import nn
from torch_geometric.nn import pool

from models.orientability.BaseOrientability import BaseOrientabilityModule


class SCNNNetwork(BaseOrientabilityModule):
    def __init__(
        self,
        rank,
        in_channels,
        hidden_channels,
        out_channels,
        conv_order_down,
        conv_order_up,
        n_layers=3,
    ):
        super().__init__()
        self.rank = rank
        self.base_model = SCNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            conv_order_down=conv_order_down,
            conv_order_up=conv_order_up,
            n_layers=n_layers,
        )
        self.liner_readout = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, laplacian_down, laplacian_up, signal_belongings):
        x = self.base_model(x, laplacian_down, laplacian_up)
        x = self.liner_readout(x)
        x_mean = pool.global_mean_pool(x, signal_belongings)
        x_mean[torch.isnan(x_mean)] = 0
        return x_mean

    def general_step(
        self, batch, batch_idx, step: Literal["train", "test", "validation"]
    ):
        s_complexes, signal_belongings, batch_len = batch
        x = s_complexes.signals[self.rank]
        if self.rank == 0:
            laplacian_down = None
            laplacian_up = s_complexes.neighborhood_matrices[f"0_laplacian"]
        elif self.rank == 1:
            laplacian_down = s_complexes.neighborhood_matrices[
                f"1_laplacian_down"
            ]
            laplacian_up = s_complexes.neighborhood_matrices[f"1_laplacian_up"]
        elif self.rank == 2:
            laplacian_down = s_complexes.neighborhood_matrices[f"2_laplacian"]
            laplacian_up = None
        else:
            raise ValueError("rank must be 0, 1 or 2.")
        y = s_complexes.other_features["y"].float()
        signal_belongings = signal_belongings[self.rank]
        x_hat = self(x, laplacian_down, laplacian_up, signal_belongings)
        # Squeeze x_hat to match the shape of y
        x_hat = x_hat.squeeze()
        loss = nn.functional.binary_cross_entropy_with_logits(x_hat, y)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            batch_size=batch_len,
            on_step=False,
            on_epoch=True,
        )
        self.log_accuracies(x_hat, y, batch_len, step)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.base_model.parameters(), lr=0.01)
        return optimizer
