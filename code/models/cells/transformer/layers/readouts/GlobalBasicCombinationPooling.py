import torch
from jaxtyping import Float

from models.cells.transformer.layers.readouts.BaseReadout import BaseReadout
from models.cells.transformer.layers.readouts.GlobalAddPooling import (
    GlobalAddPooling,
)
from models.cells.transformer.layers.readouts.GlobalMaxPooling import (
    GlobalMaxPooling,
)
from models.cells.transformer.layers.readouts.GlobalMeanPooling import (
    GlobalMeanPooling,
)


class GlobalBasicCombinationPooling(BaseReadout):
    def __init__(self):
        super().__init__()
        self.global_max_pooling = GlobalMaxPooling()
        self.global_add_pooling = GlobalAddPooling()
        self.global_mean_pooling = GlobalMeanPooling()
        # Initialize weights to 1 to give the same importance to each readout
        self.weight_max = torch.nn.Parameter(torch.ones(1))
        self.weight_add = torch.nn.Parameter(torch.ones(1))
        self.weight_mean = torch.nn.Parameter(torch.ones(1))

    def forward(
        self,
        x: dict[int, Float[torch.Tensor, "..."], ...],
        x_belongings: dict[int, list[int]],
    ) -> (
        dict[int, Float[torch.Tensor, "..."], ...] | Float[torch.Tensor, "..."]
    ):
        readout_result = dict()
        max_pooling_result = self.global_max_pooling(x, x_belongings)
        add_pooling_result = self.global_add_pooling(x, x_belongings)
        mean_pooling_result = self.global_mean_pooling(x, x_belongings)
        for key in x.keys():
            readout_result[key] = (
                self.weight_max * max_pooling_result[key]
                + self.weight_add * add_pooling_result[key]
                + self.weight_mean * mean_pooling_result[key]
            )
        return readout_result

    @property
    def is_global(self):
        return False
