from enum import Enum

from models.cells.transformer.WeightInitialization import WeightInitialization
from models.cells.transformer.layers.input_preprocessing.ConcatenationPositionalEncoding import (
    ConcatenationPositionalEncoding,
)
from models.cells.transformer.layers.input_preprocessing.NoPositionalEncoding import (
    NoPositionalEncoding,
)
from models.cells.transformer.layers.input_preprocessing.SumPositionalEncoding import (
    SumPositionalEncoding,
)


class InputPreprocessing(Enum):
    SUM_POSITIONAL_ENCODINGS = "sum_positional_encodings"
    NO_POSITIONAL_ENCODINGS = "no_positional_encodings"
    CONCATENATION_POSITIONAL_ENCODINGS = "concatenation_positional_encodings"


def get_input_preprocessing_layer(
    input_preprocessing_type: InputPreprocessing,
):
    match input_preprocessing_type:
        case InputPreprocessing.SUM_POSITIONAL_ENCODINGS:
            return SumPositionalEncoding
        case InputPreprocessing.NO_POSITIONAL_ENCODINGS:
            return NoPositionalEncoding
        case InputPreprocessing.CONCATENATION_POSITIONAL_ENCODINGS:
            return ConcatenationPositionalEncoding
        case _:
            raise ValueError(
                f"Positional encoding layer type {input_preprocessing_type} not recognized."
            )


def generate_input_preprocessing_layer(
    input_preprocessing_type: InputPreprocessing,
    dim_features: int,
    dim_positional_encoding: int,
    hidden_dim: int,
    initialization: WeightInitialization,
):
    input_preproccesing_layer_class = get_input_preprocessing_layer(
        input_preprocessing_type
    )
    if input_preprocessing_type == InputPreprocessing.NO_POSITIONAL_ENCODINGS:
        return input_preproccesing_layer_class(
            dim_features=dim_features,
            hidden_dim=hidden_dim,
            initialization=initialization,
        )
    else:
        return input_preproccesing_layer_class(
            dim_features=dim_features,
            dim_positional_encoding=dim_positional_encoding,
            hidden_dim=hidden_dim,
            initialization=initialization,
        )
