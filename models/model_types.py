from enum import Enum


class ModelType(Enum):
    GAT = "gat"
    GCN = "gcn"
    MLP = "mlp"
    TAG = "tag"
    TransfConv = "transfconv"
