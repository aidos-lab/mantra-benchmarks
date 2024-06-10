"""
Collection of models. Useful for quantitative comparisons and templating.
"""

from enum import Enum
from typing import Dict, Union, Annotated
from pydantic import Tag
import torch.nn as nn
from models.GCN import GCN, GCNConfig
from models.GAT import GAT, GATConfig
from models.MLP import MLP, MLPConfig
from models.TransfConv import TransfConv, TransfConvConfig
from models.TAG import TAG, TAGConfig
from .model_types import ModelType


model_lookup: Dict[ModelType, nn.Module] = {
    ModelType.GAT: GAT,
    ModelType.GCN: GCN,
    ModelType.MLP: MLP,
    ModelType.TAG: TAG,
    ModelType.TransfConv: TransfConv,
}

model_cfg_lookup: Dict[ModelType, nn.Module] = {
    ModelType.GAT: GATConfig,
    ModelType.GCN: GCNConfig,
    ModelType.MLP: MLPConfig,
    ModelType.TAG: TAGConfig,
    ModelType.TransfConv: TransfConvConfig,
}

ModelConfig = Union[
    Annotated[MLPConfig, Tag(ModelType.MLP)],
    Annotated[GATConfig, Tag(ModelType.GAT)],
    Annotated[GCNConfig, Tag(ModelType.GCN)],
    Annotated[TransfConvConfig, Tag(ModelType.TransfConv)],
    Annotated[TAGConfig, Tag(ModelType.TAG)],
]
