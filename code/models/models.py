"""
Collection of models. Useful for quantitative comparisons and templating.
"""

from typing import Dict, Union, Annotated, Callable
from pydantic import Tag
import torch.nn as nn
from torch_geometric.loader import DataLoader

from datasets.topox_dataloader import SimplicialTopoXDataloader
from models.GCN import GCN, GCNConfig
from models.GAT import GAT, GATConfig
from models.MLP import MLP, MLPConfig
from models.DECT import DECTMLP, DECTConfig
from models.TransfConv import TransfConv, TransfConvConfig
from models.TAG import TAG, TAGConfig
from .model_types import ModelType
from pydantic import BaseModel

from models.simplicial_complexes.san import SAN, SANConfig
from models.simplicial_complexes.sccn import SCCN, SCCNConfig
from models.simplicial_complexes.sccnn import SCCNN, SCCNNConfig
from models.simplicial_complexes.scn import SCN, SCNConfig

model_lookup: Dict[ModelType, nn.Module] = {
    ModelType.DECT: DECTMLP,
    ModelType.GAT: GAT,
    ModelType.GCN: GCN,
    ModelType.MLP: MLP,
    ModelType.TAG: TAG,
    ModelType.SAN: SAN,
    ModelType.SCCN: SCCN,
    ModelType.SCCNN: SCCNN,
    ModelType.SCN: SCN,
    ModelType.TransfConv: TransfConv,
}

ModelConfig = Union[
    Annotated[DECTConfig, Tag(ModelType.DECT)],
    Annotated[MLPConfig, Tag(ModelType.MLP)],
    Annotated[GATConfig, Tag(ModelType.GAT)],
    Annotated[GCNConfig, Tag(ModelType.GCN)],
    Annotated[TransfConvConfig, Tag(ModelType.TransfConv)],
    Annotated[TAGConfig, Tag(ModelType.TAG)],
    Annotated[SANConfig, Tag(ModelType.SAN)],
    Annotated[SCCNConfig, Tag(ModelType.SCCN)],
    Annotated[SCCNNConfig, Tag(ModelType.SCCNN)],
    Annotated[SCNConfig, Tag(ModelType.SCN)],
]

model_cfg_lookup: Dict[ModelType, ModelConfig] = {
    ModelType.DECT: DECTConfig,
    ModelType.GAT: GATConfig,
    ModelType.GCN: GCNConfig,
    ModelType.MLP: MLPConfig,
    ModelType.TAG: TAGConfig,
    ModelType.SAN: SANConfig,
    ModelType.SCCN: SCCNConfig,
    ModelType.SCCNN: SCCNNConfig,
    ModelType.SCN: SCNConfig,
    ModelType.TransfConv: TransfConvConfig,
}

dataloader_lookup: Dict[ModelType, Callable] = {
    ModelType.DECT: DataLoader,
    ModelType.GAT: DataLoader,
    ModelType.GCN: DataLoader,
    ModelType.MLP: DataLoader,
    ModelType.TAG: DataLoader,
    ModelType.SAN: SimplicialTopoXDataloader,
    ModelType.SCCN: SimplicialTopoXDataloader,
    ModelType.SCCNN: SimplicialTopoXDataloader,
    ModelType.SCN: SimplicialTopoXDataloader,
    ModelType.TransfConv: DataLoader,
}
