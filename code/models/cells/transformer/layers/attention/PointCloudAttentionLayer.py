from itertools import product

import dgl.sparse as dglsp
import torch
from dgl.sparse import SparseMatrix
from jaxtyping import Float
from torch import nn

from models.cells.transformer.TensorDiagram import Interaction
from models.cells.transformer.WeightInitialization import WeightInitialization
from models.cells.transformer.layers.attention.MaskType import MaskType
from models.cells.transformer.layers.attention.PairwiseAttentionLayer import (
    PairwiseAttentionLayer,
)


class PointCloudAttentionLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        use_bias: bool = True,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        drop_path_probability: float = 0.0,
        initialization: WeightInitialization = WeightInitialization.XAVIER_UNIFORM,
        transformer_mlp_embedding_dim_multiplier: int = 2,
        attention_mask_type: MaskType = MaskType.NO_MASK,
    ):
        super().__init__()
        # Parameters of the layer
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_bias = use_bias
        # Normalization layer.
        self.layer_norm = nn.LayerNorm(hidden_size)
        # Attention layer
        self.attention_layer = PairwiseAttentionLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            drop_path_probability=drop_path_probability,
            mlp_embedding_dim_multiplier=transformer_mlp_embedding_dim_multiplier,
            initialization=initialization,
            use_bias=use_bias,
            attention_mask_type=attention_mask_type,
        )

    def forward(
        self,
        x: Float[torch.Tensor, "..."],
        batch_mask: SparseMatrix,
        attention_mask: SparseMatrix,
    ):
        # First, we get normalized signals for the concatenation of signals x.
        normalized_x = self.layer_norm(x)
        # Then, we apply the attention layer.
        return self.attention_layer(
            h_target=x,
            h_source_normalized=normalized_x,
            h_target_normalized=normalized_x,
            batch_mask=batch_mask,
            attention_mask=attention_mask,
        )

    @staticmethod
    def _get_joint_mask(
        masks: dict[Interaction, SparseMatrix],
        dims: list[int],
        n_samples: dict[int, int],
    ) -> SparseMatrix:
        # We iterate over all the pairs of dimensions to get the joint batch mask.
        indices_to_concat = []
        values_to_concat = []
        for i, j in list(product(dims, repeat=2)):
            # For each interaction, we generate a submatrix of the complete batch masks by translating the coefficients
            # to its corresponding part.
            interaction = Interaction(i, j)
            batch_mask = masks[interaction]
            bm_indices = batch_mask.indices()
            bm_values = batch_mask.val
            # We translate the indices to the corresponding part of the complete batch mask.
            indices = torch.stack(
                [
                    bm_indices[0]
                    + sum(n_samples[d] for d in dims[:j]),  # Out dim are rows
                    bm_indices[1] + sum(n_samples[d] for d in dims[:i]),
                ],
                dim=0,
            )  # In dim are columns
            indices_to_concat.append(indices)
            values_to_concat.append(bm_values)
        # Concatenate indices and values to generate the sparse matrix
        indices = torch.cat(indices_to_concat, dim=1)
        values = torch.cat(values_to_concat, dim=0)
        total_samples = sum(n_samples.values())
        return dglsp.spmatrix(
            indices, values, shape=(total_samples, total_samples)
        )

    @staticmethod
    def get_joint_signals_and_batch_and_attention_matrices(
        x: dict[int, Float[torch.Tensor, "..."]],
        batch_masks: dict[Interaction, SparseMatrix],
        attention_masks: dict[Interaction, SparseMatrix],
    ) -> tuple[Float[torch.Tensor, "..."], SparseMatrix, SparseMatrix]:
        # First, we get dimensions that we will concatenate from x
        dims = list(x.keys())
        n_samples = {dim: x[dim].shape[0] for dim in dims}
        # We concatenate the signals in the order given by dims. We assume that the signals of dim dims[i] have shape
        # [n_samples[i], hidden_size] and we concatenate them along the first dimension to have a tensor of shape
        # [n_samples_[0] + n_samples[1], + ... + n_samples[n], hidden_size], where n_sample[i] represents
        # x[dim].shape[0] that is, the number of signals of dimension dims[i].
        x_concatenated = torch.cat([x[dim] for dim in dims], dim=0)
        # Now we get the attention and batch masks for the concatenated signals.
        batch_mask = PointCloudAttentionLayer._get_joint_mask(
            batch_masks, dims, n_samples
        )
        attention_mask = PointCloudAttentionLayer._get_joint_mask(
            attention_masks, dims, n_samples
        )
        return x_concatenated, batch_mask, attention_mask

    @staticmethod
    def disentangle_signals(
        reference_disentangled_x: dict[int, Float[torch.Tensor, "..."]],
        x: Float[torch.Tensor, "..."],
    ) -> dict[int, Float[torch.Tensor, "..."]]:
        """
        Disentangles the signals of the concatenated tensor x into signals arranged as reference_disentangled_x.
        It is assumed that x was first obtained by calling the method
        get_joint_signals_and_batch_and_attention_matrices with x equal to reference_disentangled_x.
        :param reference_disentangled_x: Dictionary of signals with the structure that the disentangled x should follow.
        :param x: Entangled signals. May be processed or not after using
        get_joint_signals_and_batch_and_attention_matrices
        :return: Dictionary of disentangled signals.
        """
        # First, we get dimensions that we will use to disentangle x. As keys() returns a stable view, this means
        # that when converting keys() to list we obtain the same list as the one we obtained in
        # get_joint_signals_and_batch_and_attention_matrices when we first entangled reference_disentangled_x
        dims = list(reference_disentangled_x.keys())
        n_samples = {
            dim: reference_disentangled_x[dim].shape[0] for dim in dims
        }
        # We disentangle the signals by splitting the tensor x into the signals of the different dimensions.
        disentangled_x = dict()
        start = 0
        for dim in dims:
            n_samples_dim = n_samples[dim]
            disentangled_x[dim] = x[start : start + n_samples_dim]
            start += n_samples_dim
        return disentangled_x
