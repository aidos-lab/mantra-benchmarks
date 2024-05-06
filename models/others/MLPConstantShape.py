import torch.nn as nn
from torch_geometric.nn import pool


class MLPConstantShape(nn.Module):
    def __init__(
        self,
        num_input_neurons,
        num_hidden_neurons,
        num_hidden_layers,
        num_out_neurons,
    ):
        super().__init__()
        self.input_layer = nn.Linear(num_input_neurons, num_hidden_neurons)
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(num_hidden_neurons, num_hidden_neurons)
                for _ in range(num_hidden_layers)
            ]
        )
        self.output_layer = nn.Linear(num_hidden_neurons, num_out_neurons)

    def forward(self, x, signal_belongings):
        x = self.input_layer(x)
        x = nn.functional.relu(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = nn.functional.relu(x)
        x = self.output_layer(x)
        return pool.global_mean_pool(x, signal_belongings)
