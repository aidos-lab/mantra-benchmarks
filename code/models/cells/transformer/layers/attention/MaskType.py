from enum import Enum


class MaskType(Enum):
    """
    Enum class for the different types of masks that can be used in the attention mechanism before sparse softmax.
    The SUM mask adds the mask to the attention coefficients, the PRODUCT mask multiplies the mask with the attention
    coefficients, and the NO_MASK type does not apply any mask.
    """

    SUM = "sum"
    PRODUCT = "product"
    NO_MASK = "none"
