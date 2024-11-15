import torch
from jaxtyping import Float

from models.cells.transformer.layers.readouts.BaseReadout import BaseReadout


class NoReadout(BaseReadout):
    def __init__(self, only_faces: bool = False):
        super().__init__()
        self.only_faces = only_faces

    def forward(
        self,
        x: dict[int, Float[torch.Tensor, "..."], ...],
        x_belongings: dict[int, list[int]],
    ) -> (
        dict[int, Float[torch.Tensor, "..."], ...] | Float[torch.Tensor, "..."]
    ):
        if self.only_faces:
            return x[2]
        return x

    @property
    def is_global(self):
        return True
