import torch
import torch_geometric.nn as pyg_nn
from jaxtyping import Float

from models.cells.transformer.layers.readouts.BaseReadout import BaseReadout


class Set2SetPooling(BaseReadout):
    def __init__(self, input_dim: int, num_iterations: int = 2):
        super().__init__()
        self.num_iterations = num_iterations
        self.input_dim = input_dim
        self.set2set = pyg_nn.Set2Set(
            in_channels=input_dim, processing_steps=num_iterations
        )

    @property
    def output_dim(self):
        return self.num_iterations * self.input_dim

    def forward(
        self,
        x: dict[int, Float[torch.Tensor, "..."], ...],
        x_belongings: dict[int, list[int]],
    ) -> (
        dict[int, Float[torch.Tensor, "..."], ...] | Float[torch.Tensor, "..."]
    ):
        # Concatenate all x tensors
        sorted_x_indices = sorted(list(x.keys()))
        x_concat = torch.cat([x[i] for i in sorted_x_indices], dim=0)
        x_belongings_concat = torch.cat(
            [x_belongings[i] for i in sorted_x_indices], dim=0
        )
        out = self.set2set(x=x_concat, index=x_belongings_concat)
        return out

    @property
    def is_global(self):
        return True
