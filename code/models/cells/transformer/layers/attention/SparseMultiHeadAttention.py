from typing import Optional

import dgl.sparse as dglsp
import torch
from dgl.sparse import SparseMatrix
from jaxtyping import Float
from torch import nn

from general_utils import generate_repeated_sparse_matrix
from models.cells.transformer.WeightInitialization import (
    WeightInitialization,
    get_initialization_function,
)
from models.cells.transformer.layers.attention.MaskType import MaskType


class SparseMultiHeadAttention(nn.Module):
    """
    Sparse Multi-Head Attention layer for implementing general transformers.
    This layer is designed to work with sparse attention and batch masks and the library DGL for fast inference.
    :parameter
    hidden_size: int - Input and output sizes of the input and target signals.
    num_heads: int - Number of attention heads. The size for each head is hidden_size // num_heads.
    use_bias: bool - Whether to use bias in the projection layers.
    dropout: float - Dropout rate for the attention coefficients before applying softmax.
    attention_mask_type: MaskType - Type of mask to apply to the attention coefficients. Options are NO_MASK, SUM,
    and PRODUCT.

    - NO_MASK: No attention mask is applied to the attention coefficients.
    - SUM: The attention mask is added to the attention coefficients for each head.
    - PRODUCT: The attention mask is multiplied to the attention coefficients element-wise for each head. If
               PRODUCT is selected, the batch_mask of the forward method is ignored.
    """

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            use_bias: bool = True,
            dropout: float = 0.0,
            attention_mask_type: MaskType = MaskType.NO_MASK,
            initialization: WeightInitialization = WeightInitialization.XAVIER_UNIFORM,
    ):
        super().__init__()
        # Parameters of the MultiHeadAttention
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.attention_mask_type = attention_mask_type
        # Projection layers
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        # Dropout
        self.dropout = dropout

        # Mask type. For SUM, we have a weight parameter to control the importance of the mask.
        if self.attention_mask_type == MaskType.SUM:
            self.bias_importance = nn.Parameter(torch.tensor(1.0))

        # Initialization
        self.initialization = initialization
        self.reset_parameters()

    def reset_parameters(self, gain: float = 0.7071):
        init_fn = get_initialization_function(self.initialization, gain)
        init_fn(self.q_proj.weight)
        init_fn(self.k_proj.weight)
        init_fn(self.v_proj.weight)
        init_fn(self.out_proj.weight)

    def apply_dropout_to_attention_coefficients(
            self, attention_coefficients: SparseMatrix
    ) -> SparseMatrix:
        """
        Applies dropout to the attention coefficients.
        :param attention_coefficients: Attention coefficients. Shape (n_target, n_source, nh)
        :return:
        """
        if self.dropout > 0.0:
            values_with_dropout = nn.functional.dropout(
                attention_coefficients.val, p=self.dropout, training=self.training
            )
            attention_coefficients_d = dglsp.spmatrix(
                attention_coefficients.indices(),
                values_with_dropout,
                attention_coefficients.shape,
            )
            return attention_coefficients_d
        return attention_coefficients

    def forward(
            self,
            h_source: Float[torch.Tensor, "n_source dh  nh"],
            h_target: Float[torch.Tensor, "n_target dh nh"],
            batch_mask: Optional[SparseMatrix] = None,
            attention_mask: Optional[SparseMatrix] = None,
    ) -> Float[torch.Tensor, "n_target dh nh"]:
        """
        Performs multi-head attention with sparse masks for batches and attention.
        Batch mask is used to allow only attention between signals of the same batch.
        To do this, a SparseMatrix B of size (n_source, n_target) where B[i, j] = 1
        if i-th source signal and j-th target signal are in the same batch and
        B[i, j] = 0 otherwise.
        Attention mask is used as a explicit bias in the attention coefficients. Concretely,
        the attention mask is a SparseMatrix A of size (n_source, n_target)
        modeling relationships between source and target signals. Incidence or adjacency matrices
        are examples of attention masks. Attention masks are either summed, multiplied, or not applied to the
        attention coefficients element-wise depending on the self.mask_type value.
        If masks are not present (= None), usual attention is performed. In the case of batch_mask,
        Q*K^T is performed using standard torch dense operations. In the case of attention_mask,
        no masks are applied to the attention coefficients.
        Remark: We assume that, if attention_mask is present and mask_type is product, the attention mask
        handles the batch mask as well, and we ignore any batch_mask passed.
        :param h_source: Source signals. Shape (n_source, dh, nh), dh = head_dim, nh = num_heads
        :param h_target: Target signals. Shape (n_target, dh, nh), dh = head_dim, nh = num_heads
        :param batch_mask: Mask to allow only attention between signals of the same batch. Shape (n_target, n_source)
        :param attention_mask: Mask to bias the attention coefficients. Shape (n_target, n_source)
        :return: Attention output. Shape (n_target, dh, nh), dh = head_dim, nh = num_heads
        """
        n_source = h_source.shape[0]
        n_target = h_target.shape[0]
        # First, we obtain query, keys, and values from our source and target signals.
        # [n_target, dh, nh], dh = head_dim, nh = num_heads
        q = self.q_proj(h_target).reshape(n_target, self.head_dim, self.num_heads)
        q *= self.scale
        # [n_source, dh, nh]
        k = self.k_proj(h_source).reshape(n_source, self.head_dim, self.num_heads)
        v = self.v_proj(h_source).reshape(n_source, self.head_dim, self.num_heads)
        # Perform attention using masks.
        k_transposed = torch.transpose(k, 0, 1)
        if self.attention_mask_type == MaskType.PRODUCT:
            # In the case we have a product mask type, we simply perform the sparse product of q and k_transposed
            # using the attention_mask and ignoring the batch_mask.
            not_normalized_attn = dglsp.bsddmm(attention_mask, q, k_transposed)
        else:
            # Otherwise, we use the batch mask, if available, and then we apply the attention mask
            not_normalized_attn = (
                dglsp.bsddmm(batch_mask, q, k_transposed)
                if batch_mask is not None
                else torch.matmul(q, k_transposed)
            )
        if self.attention_mask_type == MaskType.SUM:
            # If we have a SUM mask type, we add the attention mask multiplied by the bias_importance parameter.
            # We perform this addition on each head.
            not_normalized_attn = dglsp.add(
                not_normalized_attn,
                generate_repeated_sparse_matrix(
                    self.bias_importance * attention_mask, self.num_heads
                ),
            )
        # Normalize the attention coefficients
        attn = dglsp.softmax(not_normalized_attn, dim=1)
        # Apply dropout after the softmax and multiply by the values.
        attn = self.apply_dropout_to_attention_coefficients(attn)
        out = dglsp.bspmm(attn, v)
        # Finally, perform the output projection and return
        return self.out_proj(out.reshape(n_target, -1))
