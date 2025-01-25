from enum import Enum
from functools import partial

from torch import nn


class WeightInitialization(Enum):
    XAVIER_UNIFORM = "xavier_uniform"
    XAVIER_NORMAL = "xavier_normal"


def get_initialization_function(
    initialization: WeightInitialization, gain: float = 1.0
):
    match initialization:
        case WeightInitialization.XAVIER_UNIFORM:
            init_fn = partial(nn.init.xavier_uniform_, gain=gain)
        case WeightInitialization.XAVIER_NORMAL:
            init_fn = partial(nn.init.xavier_normal_, gain=gain)
        case _:
            raise RuntimeError("Initialization method not recognized.")
    return init_fn
