import torch
import torch_geometric.nn as pyg_nn
from jaxtyping import Float

from models.cells.transformer.layers.readouts.BaseReadout import BaseReadout


class AllGlobalAddPooling(BaseReadout):
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
        out = pyg_nn.global_add_pool(x_concat, x_belongings_concat)
        return out

    @property
    def is_global(self):
        return True
