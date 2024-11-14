"""
Collection of models. Useful for quantitative comparisons and templating.
"""

from typing import Dict, Union, Annotated, Callable

import torch.nn as nn
from pydantic import Tag
from torch_geometric.loader import DataLoader

from datasets.topox_dataloader import SimplicialTopoXDataloader
from datasets.transformer_dataloader import TransformerDataloader
from models.GAT import GAT, GATConfig
from models.TAG import TAG, TAGConfig
from models.TransfConv import TransfConv, TransfConvConfig
from models.simplicial_complexes.san import SAN, SANConfig
from models.simplicial_complexes.sccn import SCCN, SCCNConfig
from models.simplicial_complexes.sccnn import SCCNN, SCCNNConfig
from models.simplicial_complexes.scn import SCN, SCNConfig
from .GCN import GCN, GCNConfig
from .MLP import MLP, MLPConfig
from .cells.transformer.CellularTransformer import CellularTransformer, CellularTransformerConfig
from .model_types import ModelType

model_lookup: Dict[ModelType, nn.Module] = {
    ModelType.GAT: GAT,
    ModelType.GCN: GCN,
    ModelType.MLP: MLP,
    ModelType.TAG: TAG,
    ModelType.SAN: SAN,
    ModelType.SCCN: SCCN,
    ModelType.SCCNN: SCCNN,
    ModelType.SCN: SCN,
    ModelType.TransfConv: TransfConv,
    ModelType.CELL_TRANSF: CellularTransformer
}

ModelConfig = Union[
    Annotated[MLPConfig, Tag(ModelType.MLP)],
    Annotated[GATConfig, Tag(ModelType.GAT)],
    Annotated[GCNConfig, Tag(ModelType.GCN)],
    Annotated[TransfConvConfig, Tag(ModelType.TransfConv)],
    Annotated[TAGConfig, Tag(ModelType.TAG)],
    Annotated[SANConfig, Tag(ModelType.SAN)],
    Annotated[SCCNConfig, Tag(ModelType.SCCN)],
    Annotated[SCCNNConfig, Tag(ModelType.SCCNN)],
    Annotated[SCNConfig, Tag(ModelType.SCN)],
    Annotated[CellularTransformerConfig, Tag(ModelType.CELL_TRANSF)]
]

model_cfg_lookup: Dict[ModelType, ModelConfig] = {
    ModelType.GAT: GATConfig,
    ModelType.GCN: GCNConfig,
    ModelType.MLP: MLPConfig,
    ModelType.TAG: TAGConfig,
    ModelType.SAN: SANConfig,
    ModelType.SCCN: SCCNConfig,
    ModelType.SCCNN: SCCNNConfig,
    ModelType.SCN: SCNConfig,
    ModelType.TransfConv: TransfConvConfig,
    ModelType.CELL_TRANSF: CellularTransformerConfig
}

dataloader_lookup: Dict[ModelType, Callable] = {
    ModelType.GAT: DataLoader,
    ModelType.GCN: DataLoader,
    ModelType.MLP: DataLoader,
    ModelType.TAG: DataLoader,
    ModelType.SAN: SimplicialTopoXDataloader,
    ModelType.SCCN: SimplicialTopoXDataloader,
    ModelType.SCCNN: SimplicialTopoXDataloader,
    ModelType.SCN: SimplicialTopoXDataloader,
    ModelType.TransfConv: DataLoader,
    ModelType.CELL_TRANSF: TransformerDataloader,
}
