import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GCNetwork(nn.Module):
    def __init__(
        self,
        hidden_channels,
        num_node_features,
        out_channels,
        num_hidden_layers,
    ):
        super().__init__()
        self.conv_input = GCNConv(num_node_features, hidden_channels)
        self.hidden_layers = nn.ModuleList(
            [
                GCNConv(hidden_channels, hidden_channels)
                for _ in range(num_hidden_layers)
            ]
        )
        self.final_linear = nn.Linear(hidden_channels, out_channels)

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
