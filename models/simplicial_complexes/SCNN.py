import torch
from topomodelx.nn.simplicial.scnn import SCNN
from torch import nn
from torch_geometric.nn import pool


class SCNNNetwork(nn.Module):
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
