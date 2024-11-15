import itertools
from typing import Optional, Dict, List

import dgl.sparse as dglsp
import torch
from dgl.sparse import SparseMatrix
from jaxtyping import Float, Int
from pydantic import BaseModel
from torch import nn

from math_utils import sparse_abs
from ...model_types import ModelType
from models.cells.transformer.DataTypes import (
    NeighborhoodMatrixType,
    NeighborhoodType,
    CellComplexData,
)
from models.cells.transformer.InputPreprocessing import (
    InputPreprocessing,
    generate_input_preprocessing_layer,
)
from models.cells.transformer.Readout import Readout, get_readout_layer
from models.cells.transformer.TensorDiagram import TensorDiagram, Interaction
from models.cells.transformer.WeightInitialization import WeightInitialization
from models.cells.transformer.layers.BottleneckMLP import BottleneckMLP
from models.cells.transformer.layers.attention.HierarchicalAttentionLayer import (
    HierarchicalAttentionLayer,
)
from models.cells.transformer.layers.attention.MaskType import MaskType
from models.cells.transformer.layers.attention.PointCloudAttentionLayer import (
    PointCloudAttentionLayer,
)
from models.cells.transformer.positional_encodings.PositionalEncodings import (
    PositionalEncodingsType,
)


def _get_batch_mask_improved(
    s: Float[torch.Tensor, "1 source_simplices"],
    t: Float[torch.Tensor, "1 target_simplices"],
) -> Float[torch.Tensor, "target_simplices source_simplices"]:
    mask_n = torch.zeros((len(t), len(s)), dtype=torch.float32)
    ps_start, pt_start = 0, 0
    ps, pt = 0, 0

    while ps < len(s) or pt < len(t):
        if ps < len(s) and pt < len(t) and s[ps].item() == t[pt].item():
            key = s[ps].item()
            while ps < len(s) and s[ps].item() == key:
                ps += 1
            while pt < len(t) and t[pt].item() == key:
                pt += 1
            mask_n[pt_start:pt, ps_start:ps] = 1
            ps_start, pt_start = ps, pt

        elif ps < len(s) and (pt >= len(t) or s[ps].item() < t[pt].item()):
            key = s[ps].item()
            while ps < len(s) and s[ps].item() == key:
                ps += 1
                ps_start += 1
        elif pt < len(t) and (ps >= len(s) or t[pt].item() < s[ps].item()):
            key = t[pt].item()
            while pt < len(t) and t[pt].item() == key:
                pt += 1
                pt_start += 1

        if ps >= len(s) and pt >= len(t):
            break

    if ps_start < len(s) and pt_start < len(t):
        mask_n[pt_start:pt, ps_start:ps] = 1

    return mask_n


def _get_batch_mask(
    x: CellComplexData, interaction: Interaction, improved: bool = False
):
    target_belongings = x.other_features["signals_belonging"][
        interaction.out_dim
    ]
    source_belongings = x.other_features["signals_belonging"][
        interaction.in_dim
    ]
    if improved:
        batch_mask_dense = _get_batch_mask_improved(
            s=source_belongings, t=target_belongings
        ).to_sparse()
    else:
        batch_mask_dense = (
            target_belongings.unsqueeze(1) == source_belongings
        ).to_sparse()
    batch_mask = dglsp.spmatrix(
        indices=batch_mask_dense.indices().to(target_belongings.device),
        shape=tuple(batch_mask_dense.shape),
    )
    return batch_mask


def _get_attention_mask(x: CellComplexData, interaction: Interaction):
    # Let i, j be the input and output dimensions of the interaction. Five possibilities possibilities:
    # 1. i = j: The attention mask is given by adjacency if i=j=0, and by coadjacency if i=j>0.
    # 2. i = j-1: The attention mask is given by the transpose of the incidence matrix from j to i.
    # 3. i = j+1: The attention mask is given by the incidence matrix from i to j.
    # 4. |i-j| > 1: The attention matrix is the zero matrix.
    if interaction.in_dim == interaction.out_dim:
        if interaction.in_dim == 0:
            neighborhood_type = NeighborhoodMatrixType(
                type=NeighborhoodType.UPPER_ADJACENCY, dimension=0
            )
            attention_masks = x.neighborhood_matrices[neighborhood_type]
        else:
            neighborhood_type = NeighborhoodMatrixType(
                type=NeighborhoodType.LOWER_ADJACENCY,
                dimension=interaction.in_dim,
            )
            attention_masks = x.neighborhood_matrices[neighborhood_type]
    elif interaction.in_dim == interaction.out_dim - 1:
        neighborhood_type = NeighborhoodMatrixType(
            type=NeighborhoodType.BOUNDARY, dimension=interaction.out_dim
        )
        attention_masks = sparse_abs(
            x.neighborhood_matrices[neighborhood_type].transpose()
        )
    elif interaction.in_dim == interaction.out_dim + 1:
        neighborhood_type = NeighborhoodMatrixType(
            type=NeighborhoodType.BOUNDARY, dimension=interaction.in_dim
        )
        attention_masks = sparse_abs(
            x.neighborhood_matrices[neighborhood_type]
        )
    else:
        attention_masks = dglsp.spmatrix(
            indices=torch.tensor([[0], [0]]).to(x.signals[0].device),
            val=torch.tensor([0.0]).to(x.signals[0].device),
            shape=(
                x.signals[interaction.out_dim].shape[0],
                x.signals[interaction.in_dim].shape[0],
            ),
        )
    return attention_masks


def _get_layer_mask_types(
    attention_mask_types: Optional[list[MaskType, ...] | MaskType],
    number_of_layers: int,
) -> list[MaskType, ...]:
    if attention_mask_types is None:
        attention_mask_types = [
            MaskType.NO_MASK for _ in range(number_of_layers)
        ]
    else:
        if not isinstance(attention_mask_types, list):
            # If only one mask type is provided, we repeat it for all layers.
            attention_mask_types = [
                attention_mask_types for _ in range(number_of_layers)
            ]
    assert len(attention_mask_types) == number_of_layers, (
        "The number of attention mask types must be equal "
        "to the number of layers."
    )
    return attention_mask_types


def _get_layer_tensor_diagrams(
    tensor_diagram_input: Optional[list[TensorDiagram, ...] | TensorDiagram],
    input_sizes: dict[int, int],
    number_of_layers: int,
) -> list[TensorDiagram, ...]:
    if tensor_diagram_input is None:
        # In case tensor diagram(s) is (are) not provided, we use a fully connected tensor diagram.
        # First we get all the pairs of keys in the input sizes
        keys = list(input_sizes.keys())
        # Then, the tensor diagram string contains all possible interactions, that is equivalent to the interactions
        # given by the cartesian product of the keys with themselves.
        tensor_diag_str = ",".join(
            [
                f"{in_dim}->{out_dim}"
                for in_dim, out_dim in itertools.product(keys, keys)
            ]
        )
        tensor_diagram = TensorDiagram(tensor_diag_str)
        tensor_diagram_input = [
            tensor_diagram for _ in range(number_of_layers)
        ]
    else:
        if not isinstance(tensor_diagram_input, list):
            # If only one tensor diagram is provided, we repeat it for all layers.
            tensor_diagram_input = [
                tensor_diagram_input for _ in range(number_of_layers)
            ]

    assert len(tensor_diagram_input) == number_of_layers, (
        "The number of tensor diagrams must be equal "
        "to the number of layers."
    )
    return tensor_diagram_input


def _get_input_preprocessing_layers(
    input_preprocessing_type: (
        dict[int, InputPreprocessing] | InputPreprocessing
    ),
    input_sizes: dict[int, int],
    positional_encodings_lengths: dict[int, int],
    hidden_size: int,
    initialization: WeightInitialization,
) -> nn.ModuleList:
    if isinstance(input_preprocessing_type, dict):
        assert set(input_preprocessing_type.keys()) == set(
            input_sizes.keys()
        ), (
            "The keys of the input preprocessing type dictionary must be the same as the keys of the input sizes"
            " dictionary."
        )
    else:
        input_preprocessing_type = [
            input_preprocessing_type for _ in input_sizes
        ]
    input_preprocessing_layers = nn.ModuleList(
        [
            generate_input_preprocessing_layer(
                input_preprocessing_type=input_preprocessing_type[dim],
                dim_features=input_sizes[dim],
                dim_positional_encoding=positional_encodings_lengths[dim],
                hidden_dim=hidden_size,
                initialization=initialization,
            )
            for dim in input_sizes
        ]
    )
    return input_preprocessing_layers


class CellularTransformerConfig(BaseModel):
    type: ModelType = ModelType.CELL_TRANSF
    input_sizes: Dict[int, int]
    positional_encodings_lengths: Dict[int, int]
    out_size: int
    num_layers: int = 2
    hidden_size: int = 64
    num_heads: int = 8
    dropout_attention: float = 0.0
    dropout_activations: float = 0.0
    dropout_final_mlp: float = 0.0
    dropout_input_projections: float = 0.0
    drop_path_probability: float = 0.0
    num_hidden_layers_last_mlp: int = 2
    use_bias: bool = True
    forget_dimensions: bool = (
        False  # If True, layer_tensor_diagrams must be None and attention
    )
    # is performed with cells as points in a point cloud, forgetting the dimensions.


class CellularTransformer(nn.Module):

    def __init__(self, config: CellularTransformerConfig):
        super().__init__()
        self.input_sizes = config.input_sizes
        self.readout = Readout.GLOBAL_MEAN_POOLING  # Fixed
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.positional_encoding_type = (
            PositionalEncodingsType.HODGE_LAPLACIAN_EIGENVECTORS
        )  # Fixed
        self.input_preprocessing_type = (
            InputPreprocessing.SUM_POSITIONAL_ENCODINGS
        )  # Fixed
        self.dropout_attention = config.dropout_attention
        self.dropout_activations = config.dropout_activations
        self.drop_path_probability = config.drop_path_probability
        self.num_hidden_layers_last_mlp = config.num_hidden_layers_last_mlp
        self.use_bias = config.use_bias
        self.hidden_size = config.hidden_size
        self.initialization = WeightInitialization.XAVIER_UNIFORM  # Fixed.
        self.dropout_final_mlp = config.dropout_final_mlp
        self.out_size = config.out_size
        self.layer_tensor_diagrams = TensorDiagram(
            "0->0,0->1,1->0,1->1,1->2,2->1,2->2"
        )  # Fixed
        attention_mask_types = MaskType.SUM  # Fixed
        # Configuring the attention type. Here, we decide if we drop the hierarchical structure of the
        # cellular_data complex to perform attention.
        self.forget_dimensions = config.forget_dimensions
        if self.forget_dimensions and self.layer_tensor_diagrams is not None:
            raise ValueError(
                "If forget_dimensions is True, layer_tensor_diagrams must be None."
            )
        if self.forget_dimensions:
            layer_tensor_diagrams = None  # We use a fully connected tensor diagram, that is generated when
            # layer tensor diagrams is None.
        self.layer_tensor_diagrams = _get_layer_tensor_diagrams(
            self.layer_tensor_diagrams, config.input_sizes, config.num_layers
        )
        self.attention_mask_types = _get_layer_mask_types(
            attention_mask_types, config.num_layers
        )
        # Input preprocessing layers
        self.preproc_layers = _get_input_preprocessing_layers(
            self.input_preprocessing_type,
            config.input_sizes,
            config.positional_encodings_lengths,
            config.hidden_size,
            self.initialization,
        )
        # If the positional encodings are of eigenvector type, we randomly flip the sign of the positional encodings
        # each time we forward pass the input with the objective of learning invariance to the sign of the eigenvectors.
        self.flip_sign_pe = self.positional_encoding_type in [
            PositionalEncodingsType.HODGE_LAPLACIAN_EIGENVECTORS,
            PositionalEncodingsType.BARYCENTRIC_SUBDIVISION_GRAPH_EIGENVECTORS,
        ]
        self.input_dropout_layers = nn.ModuleList(
            [
                nn.Dropout(config.dropout_input_projections)
                for _ in config.input_sizes.keys()
            ]
        )
        # Attention layers
        self.attention_layers = self.get_attention_layers()
        if self.readout == Readout.SET2SET_POOLING:
            self.readout_layer = get_readout_layer(
                self.readout, input_dim=config.hidden_size
            )
        else:
            self.readout_layer = get_readout_layer(self.readout)
        # Final MLP block and readout layer
        self.predictor_head = self.get_prediction_head()

    def get_prediction_head(self) -> Optional[nn.ModuleDict | nn.Module]:
        in_features = (
            self.hidden_size
            if self.readout != Readout.SET2SET_POOLING
            else self.readout_layer.output_dim
        )
        if self.readout == Readout.ALL_GLOBAL_ADD_POOLING:
            return nn.ModuleDict(
                {
                    str(i): BottleneckMLP(
                        in_features=in_features,
                        out_features=self.out_size,
                        num_hidden_layers=self.num_hidden_layers_last_mlp,
                        dropout=self.dropout_final_mlp,
                    )
                    for i in self.input_sizes.keys()
                }
            )
        elif self.readout == Readout.NO_READOUT:
            return None
        else:
            return BottleneckMLP(
                in_features=in_features,
                out_features=self.out_size,
                num_hidden_layers=self.num_hidden_layers_last_mlp,
                dropout=self.dropout_final_mlp,
            )

    def get_attention_layers(self) -> nn.ModuleList:
        """
        Returns the attention layers of the transformer. If forget_dimensions is True, the attention is performed
        considering the cells as points in a point cloud, forgetting the dimensions. If forget_dimensions is False,
        the attention is performed considering the hierarchical structure of the cellular_data complex.
        These two mechanisms are implemented by the attention layers PointCloudAttentionLayer and
        HierarchicalAttentionLayer, respectively.
        :return: ModuleList containing self.num_layers attention layers of the requested type according to
        self.forget_dimensions.
        """
        if self.forget_dimensions:
            return nn.ModuleList(
                [
                    PointCloudAttentionLayer(
                        hidden_size=self.hidden_size,
                        num_heads=self.num_heads,
                        use_bias=self.use_bias,
                        attention_dropout=self.dropout_attention,
                        activation_dropout=self.dropout_activations,
                        drop_path_probability=self.drop_path_probability,
                        initialization=self.initialization,
                        attention_mask_type=self.attention_mask_types[
                            layer_idx
                        ],
                    )
                    for layer_idx in range(self.num_layers)
                ]
            )
        else:
            return nn.ModuleList(
                [
                    HierarchicalAttentionLayer(
                        hidden_size=self.hidden_size,
                        num_heads=self.num_heads,
                        use_bias=self.use_bias,
                        attention_dropout=self.dropout_attention,
                        activation_dropout=self.dropout_activations,
                        drop_path_probability=self.drop_path_probability,
                        initialization=self.initialization,
                        tensor_diagram=self.layer_tensor_diagrams[layer_idx],
                        attention_mask_type=self.attention_mask_types[
                            layer_idx
                        ],
                    )
                    for layer_idx in range(self.num_layers)
                ]
            )

    def apply_random_sign_flip_if_needed(
        self, positional_encodings: dict[int, Float[torch.Tensor, "..."]]
    ) -> dict[int, Float[torch.Tensor, "..."]]:
        with torch.no_grad():
            if self.training and self.flip_sign_pe:
                # We apply a random flip of the sign of the positional encodings only if we are training AND it is
                # needed by the nature of the positional encodings. This is indicated by the variable self.flip_sign_pe.
                updated_positional_encodings = dict()
                for dim in positional_encodings:
                    pe_dim = positional_encodings[dim]
                    rand_sign = (
                        2
                        * (
                            torch.rand(
                                pe_dim.shape[0], 1, device=pe_dim.device
                            )
                            > 0.5
                        ).float()
                        - 1.0
                    )
                    pe_dim_updated = rand_sign * pe_dim
                    updated_positional_encodings[dim] = pe_dim_updated
                return updated_positional_encodings
            else:
                return positional_encodings

    def get_attention_matrices(
        self, x: CellComplexData
    ) -> dict[Interaction, SparseMatrix]:
        attention_masks = dict()
        # We iterate over all possible interactions through the whole set of layers.
        for tensor_diagram in self.layer_tensor_diagrams:
            for interaction in tensor_diagram.interactions:
                # We avoid recomputing the attention mask if it has already been computed.
                if interaction not in attention_masks:
                    attention_masks[interaction] = _get_attention_mask(
                        x, interaction
                    )
        return attention_masks

    def get_batch_matrices(
        self, x: CellComplexData
    ) -> dict[Interaction, SparseMatrix]:
        batch_masks = dict()
        # We iterate over all possible interactions through the whole set of layers.
        for tensor_diagram in self.layer_tensor_diagrams:
            for interaction in tensor_diagram.interactions:
                # We avoid recomputing the batch mask if it has already been computed.
                if interaction not in batch_masks:
                    batch_masks[interaction] = _get_batch_mask(x, interaction)
        return batch_masks

    def forward_readout(
        self,
        h: dict[int, Float[torch.Tensor, "..."]],
        h_belongings: dict[int, Int[torch.Tensor, "total_signals"]],
    ) -> Float[torch.Tensor, "..."]:
        # Global readouts are computed using all possible features at the final attention layer.
        # No readout is a layer that do not perform any computation and that directly accepts h and
        # h_belongings as input.
        if self.readout == Readout.NO_READOUT:
            return self.readout_layer(x=h, x_belongings=h_belongings)
        if self.readout == Readout.NO_READOUT_FACES:
            h = self.readout_layer(x=h, x_belongings=h_belongings)
            out = self.predictor_head(h[2])
            return out
        if self.readout == Readout.ALL_GLOBAL_ADD_POOLING:
            h = {i: self.predictor_head[str(i)](h[i]) for i in h}
            out = self.readout_layer(x=h, x_belongings=h_belongings)
            out = out / self.num_heads
            return out
        if self.readout == Readout.SET2SET_POOLING:
            h = self.readout_layer(x=h, x_belongings=h_belongings)
            out = self.predictor_head(h)
            return out
        # Predictions and readout are computed with vertex features by convention for non global readouts,
        # because they are generally available. This can be changed by simply changing the dimension/the way the
        # output feature vectors are computed using a predictor.
        h = self.predictor_head(h[0])
        match self.readout:
            case Readout.GLOBAL_ADD_POOLING:
                out = self.readout_layer(
                    x={0: h}, x_belongings={0: h_belongings[0]}
                )
                out = (
                    out[0] / self.num_heads
                )  # We use the vertex outputs and we normalize by the number of heads.
            case Readout.GLOBAL_MEAN_POOLING:
                out = self.readout_layer(
                    x={0: h}, x_belongings={0: h_belongings[0]}
                )
                out = out[0]
                # We do not normalize in this case as the mean pooling already does it.
            case Readout.GLOBAL_MAX_POOLING:
                out = self.readout_layer(
                    x={0: h}, x_belongings={0: h_belongings[0]}
                )
                out = out[0]
                # We do not normalize in this case as the max pooling only selects the value of one vertex.
            case Readout.GLOBAL_BASIC_COMBINATION_POOLING:
                out = self.readout_layer(
                    x={0: h}, x_belongings={0: h_belongings[0]}
                )
                out = out[0]
                # We do not normalize in this case as we are learning a weight for the sum and combining with the other
                # two simple readouts mean and max.
            case _:
                raise NotImplementedError(
                    f"Readout {self.readout} is not implemented."
                )
        return out

    def forward_point_cloud_attention(
        self,
        h: dict[int, Float[torch.Tensor, "..."]],
        batch_matrices: dict[Interaction, SparseMatrix],
        attention_matrices: dict[Interaction, SparseMatrix],
    ) -> dict[int, Float[torch.Tensor, "..."]]:
        (h_joint, joint_batch_matrix, joint_attention_matrix) = (
            PointCloudAttentionLayer.get_joint_signals_and_batch_and_attention_matrices(
                x=h,
                batch_masks=batch_matrices,
                attention_masks=attention_matrices,
            )
        )
        for layer_idx in range(self.num_layers):
            h_joint = self.attention_layers[layer_idx](
                h_joint, joint_batch_matrix, joint_attention_matrix
            )
        h_updated = PointCloudAttentionLayer.disentangle_signals(
            reference_disentangled_x=h, x=h_joint
        )
        return h_updated

    def forward_hierarchical_attention(
        self,
        h: dict[int, Float[torch.Tensor, "..."]],
        batch_matrices: dict[Interaction, SparseMatrix],
        attention_matrices: dict[Interaction, SparseMatrix],
    ) -> dict[int, Float[torch.Tensor, "..."]]:
        for layer_idx in range(self.num_layers):
            h = self.attention_layers[layer_idx](
                h, batch_matrices, attention_matrices
            )
        return h

    def forward(self, x: CellComplexData) -> Float[torch.Tensor, "..."]:
        # h will contain the updated signals at each step.
        h = dict()
        # First we get the positional encodings and apply a random flip of the sign if needed.
        if (
            self.input_preprocessing_type
            == InputPreprocessing.NO_POSITIONAL_ENCODINGS
        ):
            positional_encodings = None
        else:
            positional_encodings = x.other_features["positional_encodings"]
            positional_encodings = self.apply_random_sign_flip_if_needed(
                positional_encodings
            )
        # Get attention matrices
        attention_matrices = self.get_attention_matrices(x)
        # Get batch matrices
        batch_matrices = self.get_batch_matrices(x)
        # Apply input preprocessing
        for dim in x.signals:
            pe_dim = (
                None
                if self.input_preprocessing_type
                == InputPreprocessing.NO_POSITIONAL_ENCODINGS
                else positional_encodings[dim]
            )
            h[dim] = self.preproc_layers[dim](x.signals[dim], pe_dim)
            h[dim] = self.input_dropout_layers[dim](h[dim])
        # Apply the attention layers.
        if self.forget_dimensions:
            h = self.forward_point_cloud_attention(
                h, batch_matrices, attention_matrices
            )
        else:
            h = self.forward_hierarchical_attention(
                h, batch_matrices, attention_matrices
            )
        # Perform predictions and readout.
        out = self.forward_readout(h, x.other_features["signals_belonging"])
        return out

    @property
    def label_length(self) -> int:
        return self.out_size
