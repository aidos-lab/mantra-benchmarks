from enum import Enum
from typing import Dict, Union
import torch.nn as nn
from models.GCN import GCN, GCNConfig
from models.GAT import GAT, GATConfig
from models.MLP import MLP, MLPConfig
from models.TransfConv import TransfConv, TransfConvConfig
from models.TAG import TAG, TAGConfig


class ModelType(Enum):
    GAT = "gat"
    GCN = "gcn"
    MLP = "mlp"
    TAG = "tag"
    TransfConv = "transfconv"


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
    GATConfig, GCNConfig, MLPConfig, TransfConvConfig, TAGConfig
]
