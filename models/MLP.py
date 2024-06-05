import torch.nn as nn
from torch_geometric.nn import pool
from dataclasses import dataclass
from pydantic import BaseModel


class MLPConfig(BaseModel):
    num_hidden_neurons: int = 64
    num_hidden_layers: int = 3
    num_node_features: int = 1
    out_channels: int = 5


class MLP(nn.Module):
    def __init__(
        self,
        config: MLPConfig,
    ):
        super().__init__()
        self.input_layer = nn.Linear(
            config.num_node_features, config.num_hidden_neurons
        )
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(config.num_hidden_neurons, config.num_hidden_neurons)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.output_layer = nn.Linear(
            config.num_hidden_neurons, config.out_channels
        )

    def forward(self, x, edge_index, signal_belongings):
        x = self.input_layer(x)
        x = nn.functional.relu(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = nn.functional.relu(x)
        x = self.output_layer(x)
        return pool.global_mean_pool(x, signal_belongings)
