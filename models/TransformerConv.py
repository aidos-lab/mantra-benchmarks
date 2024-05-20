import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_mean_pool, TransformerConv

from dataclasses import dataclass


@dataclass
class TransformerConvConfig:
    model_name: str = "TransformerConv"
    hidden_channels: int = 64
    num_hidden_layers: int = 3
    num_node_features: int = 1
    out_channels: int = 5


class TransformerConv(nn.Module):
    def __init__(
        self,
        config: TransformerConvConfig,
    ):
        super().__init__()
        self.conv_input = TransformerConv(
            config.num_node_features, config.hidden_channels
        )
        self.hidden_layers = nn.ModuleList(
            [
                TransformerConv(config.hidden_channels, config.hidden_channels)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.final_linear = nn.Linear(
            config.hidden_channels, config.out_channels
        )

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv_input(x, edge_index)
        for layer in self.hidden_layers:
            x = layer(x, edge_index)
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.final_linear(x)
        return x
