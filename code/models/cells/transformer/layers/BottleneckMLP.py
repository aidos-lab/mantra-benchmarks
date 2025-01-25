from torch import nn

from models.cells.transformer.WeightInitialization import (
    WeightInitialization,
    get_initialization_function,
)


class BottleneckMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_hidden_layers: int = 2,
        dropout: float = 0.0,
        initialization: WeightInitialization = WeightInitialization.XAVIER_UNIFORM,
    ):
        super().__init__()
        # Parameters
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        # MLP
        self.mlp = self._get_model()
        # Initialization
        self.initialization = initialization
        self.reset_parameters()

    def reset_parameters(self, gain: float = 1.0):
        init_fn = get_initialization_function(self.initialization, gain)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                init_fn(layer.weight)

    def _get_model(self):
        modules = []
        input_size = self.in_features
        for i in range(self.num_hidden_layers):
            modules.append(
                nn.Linear(input_size, max(self.out_features, input_size // 2))
            )
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(self.dropout))
            input_size = max(self.out_features, input_size // 2)
        modules.append(nn.Linear(input_size, self.out_features))
        return nn.Sequential(*modules)

    def forward(self, x):
        return self.mlp(x)
