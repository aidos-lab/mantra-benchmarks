import torch
import torch_geometric.nn as pyg_nn
from jaxtyping import Float

from models.cells.transformer.layers.readouts.BaseReadout import BaseReadout


class GlobalMaxPooling(BaseReadout):
    def forward(
        self,
        x: dict[int, Float[torch.Tensor, "..."], ...],
        x_belongings: dict[int, list[int]],
    ) -> (
        dict[int, Float[torch.Tensor, "..."], ...] | Float[torch.Tensor, "..."]
    ):
        readout_result = dict()
        for key in x.keys():
            readout_result[key] = pyg_nn.global_max_pool(
                x[key], x_belongings[key]
            )
        return readout_result

    @property
    def is_global(self):
        return False
