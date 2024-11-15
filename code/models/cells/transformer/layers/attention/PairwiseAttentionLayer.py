from typing import Optional

import torch
import torch.nn.functional as F
from dgl.sparse import SparseMatrix
from jaxtyping import Float
from torch import nn

from models.cells.transformer.WeightInitialization import (
    WeightInitialization,
    get_initialization_function,
)
from models.cells.transformer.layers.DropPath import DropPath
from models.cells.transformer.layers.attention.MaskType import MaskType
from models.cells.transformer.layers.attention.SparseMultiHeadAttention import (
    SparseMultiHeadAttention,
)


def _get_attending_cells(
    batch_mask: SparseMatrix,
) -> Float[torch.Tensor, "n_target"]:
    """
    Returns a tensor of size (n,) where n is the number of elements in the target sequence (rows of batch mask).
    The element i of the tensor is 1 if the element i of the target sequence is attending at least one element of the
    source sequence. Otherwise, the element i of the tensor is 0. This is done by checking if the row i in the batch
    mask has at least one non-zero element.
    :param batch_mask: Sparse batch mask matrix B. Shape (n_target, n_source). This matrix is a boolean (zero or one)
    matrix such that B[i, j] = 1 if the element i of the target sequence belongs to the same example as the element j
    of the source sequence in the batch. Otherwise, B[i, j] = 0.
    :return:
    """
    # It returns a tensor of size (n,) where n is the number of elements in the target sequence (rows of batch mask).
    # The element i of the tensor is 1 if the element i of the target sequence is attending at least one element of the
    # source sequence. Otherwise, the element i of the tensor is 0.
    return batch_mask.smin(dim=1)


def _correct_non_attending_outputs(
    h: Float[torch.Tensor, "n_target dh nh"], batch_mask: SparseMatrix
):
    """
    Corrects the outputs of the layer that are not attending anything.
    :param h: Target dim signals. Shape (n_target, dh, nh)
    :param batch_mask: Sparse batch mask. Shape (n_target, n_source)
    :return: Corrected signals. Shape (n_target, dh, nh)
    """
    signals_attending = _get_attending_cells(batch_mask)
    h = signals_attending.unsqueeze(1) * h
    return h


class PairwiseAttentionLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        mlp_embedding_dim_multiplier: int = 2,
        drop_path_probability: float = 0.0,
        initialization: WeightInitialization = WeightInitialization.XAVIER_UNIFORM,
        use_bias: bool = True,
        attention_mask_type: MaskType = MaskType.NO_MASK,
    ):
        super().__init__()
        # Dropouts
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        # Normalization layer
        self.layer_norm = nn.LayerNorm(hidden_size)
        # MultiHead Attention
        self.MHA = SparseMultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            use_bias=use_bias,
            dropout=attention_dropout,
            attention_mask_type=attention_mask_type,
            initialization=initialization,
        )
        # Feedforward layers used in the pre-norm transformer layer.
        self.FFN1 = nn.Linear(
            hidden_size,
            hidden_size * mlp_embedding_dim_multiplier,
            bias=use_bias,
        )
        self.FFN2 = nn.Linear(
            hidden_size * mlp_embedding_dim_multiplier,
            hidden_size,
            bias=use_bias,
        )
        # Drop path layer to regularize attention.
        self.drop_path_probability = drop_path_probability
        if self.drop_path_probability > 0:
            self.drop_path = DropPath(p=drop_path_probability)
        # Initialization
        self.initialization = initialization
        self.reset_parameters()

    def reset_parameters(self, gain: float = 1.0):
        init_fn = get_initialization_function(self.initialization, gain)
        init_fn(self.FFN1.weight)
        init_fn(self.FFN2.weight)

    def forward(
        self,
        h_target: Float[torch.Tensor, "n_target dh nh"],
        h_source_normalized: Float[torch.Tensor, "n_source dh nh"],
        h_target_normalized: Float[torch.Tensor, "n_target dh nh"],
        batch_mask: Optional[SparseMatrix],
        attention_mask: Optional[SparseMatrix],
    ) -> Float[torch.Tensor, "n_target dh nh"]:
        """
        Performs a pairwise pre-norm transformer layer using batch and attention masks.
        :param h_target: Unnormalized target dim signals. Shape (n_target, dh, nh)
        :param h_source_normalized: Normalized source dim signals. Shape (n_source, dh, nh)
        :param h_target_normalized: Normalized target dim signals. Shape (n_target, dh, nh)
        :param batch_mask: Sparse batch mask. Shape (n_target, n_source)
        :param attention_mask: Sparse attention mask. Shape (n_target, n_source)
        :return: Transformed signals. Shape (n_target, dh, nh)
        """
        h = self.MHA(
            h_source_normalized,
            h_target_normalized,
            batch_mask,
            attention_mask,
        )
        h = F.dropout(h, p=self.activation_dropout, training=self.training)
        if self.drop_path_probability > 0:
            h = self.drop_path(h)
        h = (
            h + h_target
        )  # Residual connection with non-normalized target signals.
        h2 = h
        h = self.layer_norm(h)
        h = F.relu(self.FFN1(h))
        h = F.dropout(h, p=self.activation_dropout, training=self.training)
        h = self.FFN2(h)
        h = F.dropout(h, p=self.activation_dropout, training=self.training)
        h = h + h2
        h = _correct_non_attending_outputs(h, batch_mask)
        return h
