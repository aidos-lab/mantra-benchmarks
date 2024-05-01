import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool


class GATNetwork(nn.Module):
    def __init__(
        self,
        hidden_channels,
        num_node_features,
        out_channels,
        num_heads,
        num_hidden_layers,
    ):
        super().__init__()
        hidden_channels_per_head = hidden_channels // num_heads
        self.gat_input = GATConv(
            num_node_features, hidden_channels_per_head, heads=num_heads
        )
        self.hidden_layers = nn.ModuleList(
            [
                GATConv(
                    hidden_channels, hidden_channels_per_head, heads=num_heads
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.final_linear = nn.Linear(hidden_channels, out_channels)

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
