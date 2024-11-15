from enum import Enum

from models.cells.transformer.layers.readouts.AllGlobalAddPooling import (
    AllGlobalAddPooling,
)
from models.cells.transformer.layers.readouts.GlobalAddPooling import (
    GlobalAddPooling,
)
from models.cells.transformer.layers.readouts.GlobalBasicCombinationPooling import (
    GlobalBasicCombinationPooling,
)
from models.cells.transformer.layers.readouts.GlobalMaxPooling import (
    GlobalMaxPooling,
)
from models.cells.transformer.layers.readouts.GlobalMeanPooling import (
    GlobalMeanPooling,
)
from models.cells.transformer.layers.readouts.NoReadout import NoReadout
from models.cells.transformer.layers.readouts.Set2SetPooling import (
    Set2SetPooling,
)


class Readout(Enum):
    GLOBAL_ADD_POOLING = "global_add_pool"
    ALL_GLOBAL_ADD_POOLING = "all_global_add_pool"
    GLOBAL_MEAN_POOLING = "global_mean_pool"
    GLOBAL_MAX_POOLING = "global_max_pool"
    NO_READOUT = "no_readout"
    NO_READOUT_FACES = "no_readout_faces"
    SET2SET_POOLING = "set2set_pool"
    GLOBAL_BASIC_COMBINATION_POOLING = "global_basic_combination_pool"


def get_readout_layer(readout_type: Readout, **kwargs):
    match readout_type:
        case Readout.GLOBAL_ADD_POOLING:
            return GlobalAddPooling()
        case Readout.GLOBAL_MEAN_POOLING:
            return GlobalMeanPooling()
        case Readout.GLOBAL_MAX_POOLING:
            return GlobalMaxPooling()
        case Readout.NO_READOUT:
            return NoReadout()
        case Readout.NO_READOUT_FACES:
            return NoReadout()
        case Readout.ALL_GLOBAL_ADD_POOLING:
            return AllGlobalAddPooling()
        case Readout.SET2SET_POOLING:
            return Set2SetPooling(input_dim=kwargs["input_dim"])
        case Readout.GLOBAL_BASIC_COMBINATION_POOLING:
            return GlobalBasicCombinationPooling()
        case _:
            raise ValueError(
                f"Readout layer type {readout_type} not recognized."
            )
