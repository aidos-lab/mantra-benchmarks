from typing import Optional

import torch
from dgl.sparse import SparseMatrix
from jaxtyping import Float
from torch import nn

from models.cells.transformer.TensorDiagram import TensorDiagram, Interaction
from models.cells.transformer.WeightInitialization import WeightInitialization
from models.cells.transformer.layers.attention.MaskType import MaskType
from models.cells.transformer.layers.attention.PairwiseAttentionLayer import (
    PairwiseAttentionLayer,
)


class HierarchicalAttentionLayer(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            use_bias: bool = True,
            attention_dropout: float = 0.0,
            activation_dropout: float = 0.0,
            drop_path_probability: float = 0.0,
            initialization: WeightInitialization = WeightInitialization.XAVIER_UNIFORM,
            tensor_diagram: Optional[TensorDiagram] = None,
            transformer_mlp_embedding_dim_multiplier: int = 2,
            attention_mask_type: MaskType = MaskType.NO_MASK,
    ):
        super().__init__()
        # Parameters of the layer
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_bias = use_bias
        self.tensor_diagram = tensor_diagram
        # Normalization layers. We have one for each dimension considered.
        self.layers_norm = nn.ModuleDict(
            {
                str(dim): nn.LayerNorm(hidden_size)
                for dim in self.tensor_diagram.considered_dimensions
            }
        )
        # Attention layers. We have one for each interaction in the tensor diagram considered.
        self.attention_layers = nn.ModuleDict(
            {
                str(interacting_dims): PairwiseAttentionLayer(
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
                for interacting_dims in self.tensor_diagram.interactions
            }
        )

    def forward(
            self,
            x: dict[int, Float[torch.Tensor, "..."]],
            batch_masks: dict[Interaction, SparseMatrix],
            attention_masks: dict[Interaction, SparseMatrix],
    ) -> dict[int, Float[torch.Tensor, "..."]]:
        """
        Performs a transformer layer using batch and attention masks following a tensor diagram that considers the
        dimension hierarchies of the signals. This is, transformer layers are computed for pairwise dimensions
        interactions in the tensor diagram.
        :param x: Dictionary of signals. The keys are the dimensions considered in the tensor diagram.
        :param batch_masks: Dictionary of batch masks for the different interactions in the tensor diagram.
        The keys are the interactions and the values are matrices of shape (n_interaction_out, n_interaction_in)
        where n_interaction_out is the number of elements in the output dimension of the interaction and
        n_interaction_in is the number of elements in the input dimension of the interaction. The batch masks are
        boolean matrices such that batch_masks[interaction][i, j] = 1 if the element i of the output dimension of the
        interaction belongs to the same example as the element j of the input dimension of the interaction. Otherwise,
        batch_masks[interaction][i, j] = 0.
        :param attention_masks: Dictionary of attention masks for the different interactions in the tensor diagram.
        The keys are the interactions and the values are matrices of shape (n_interaction_out, n_interaction_in)
        where n_interaction_out is the number of elements in the output dimension of the interaction and
        n_interaction_in is the number of elements in the input dimension of the interaction.
        A common example of attention mask is the incidence matrix of the cellular_data complex.
        :return: Dictionary of updated signals. The keys are the dimensions considered in the tensor diagram.
        """
        # The following dictionaries will store the updated signals and the accumulated signals for each dimension.
        # In particular, updated_x will be computed as the sum of the accumulated signals.
        updated_x = dict()
        accumulated_signals = {dim: list() for dim in x}
        # First, we get normalized signals for each dimension.
        normalized_x = {dim: self.layers_norm[str(dim)](x[dim]) for dim in x}
        # Secondly, we compute the transformer layer for each interaction in the tensor diagram if signals of
        # interacting dimensions are available.
        for interacting_dims, attention_layer in self.attention_layers.items():
            interaction = Interaction.from_string(interacting_dims)
            # We check if the signals of the interacting dimensions are available. If
            # they are, we compute the transformer layer for the interaction.
            if interaction.in_dim in x and interaction.out_dim in x:
                # We get the batch mask and the attention mask for the interaction.
                batch_mask = batch_masks[interaction]
                attention_mask = attention_masks[interaction]
                # We compute the transformer layer for the interaction.
                h = attention_layer(
                    h_target=x[interaction.out_dim],
                    h_source_normalized=normalized_x[interaction.in_dim],
                    h_target_normalized=normalized_x[interaction.out_dim],
                    batch_mask=batch_mask,
                    attention_mask=attention_mask,
                )
                # We accumulate the signals in the output dimension of the interaction.
                accumulated_signals[interaction.out_dim].append(h)
        # We update the signals of each dimension as the sum of the accumulated signals. If for a specific dimension
        # present at x, there are no accumulated signals, we simply discard this dimension in the updated signals.
        for dim in x:
            if len(accumulated_signals[dim]) != 0:
                updated_x[dim] = torch.mean(
                    torch.stack(accumulated_signals[dim], dim=0), dim=0
                )  # We do mean to avoid really high numbers.
        return updated_x
