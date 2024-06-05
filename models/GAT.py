import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool
from pydantic import BaseModel


class GATConfig(BaseModel):
    hidden_channels: int = 64
    num_hidden_layers: int = 3
    num_node_features: int = 1
    out_channels: int = 5
    num_heads: int = 4


class GAT(nn.Module):
    def __init__(self, config: GATConfig):
        super().__init__()
        hidden_channels_per_head = config.hidden_channels // config.num_heads
        self.gat_input = GATConv(
            config.num_node_features,
            hidden_channels_per_head,
            heads=config.num_heads,
        )
        self.hidden_layers = nn.ModuleList(
            [
                GATConv(
                    config.hidden_channels,
                    hidden_channels_per_head,
                    heads=config.num_heads,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.final_linear = nn.Linear(
            config.hidden_channels, config.out_channels
        )

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.gat_input(x, edge_index)
        for layer in self.hidden_layers:
            x = layer(x, edge_index)
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.final_linear(x)
        return x
