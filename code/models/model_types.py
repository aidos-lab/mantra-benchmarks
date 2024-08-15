from enum import Enum


class ModelType(Enum):
    SAN = "san"
    SCCN = "sccn"
    SCCNN = "sccnn"
    SCN = "scn"
    GAT = "gat"
    GCN = "gcn"
    MLP = "mlp"
    TAG = "tag"
    TransfConv = "transfconv"


model_types_str = ", ".join(type_.value for type_ in ModelType)
