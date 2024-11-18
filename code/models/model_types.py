from enum import Enum


class ModelType(Enum):
    DECT = "dect"
    SAN = "san"
    SCCN = "sccn"
    SCCNN = "sccnn"
    SCN = "scn"
    GAT = "gat"
    GCN = "gcn"
    MLP = "mlp"
    TAG = "tag"
    TransfConv = "transfconv"


simplicial_models = [
    ModelType.SAN,
    ModelType.SCCN,
    ModelType.SCCNN,
    ModelType.SCN,
]
graphbased_models = [t for t in ModelType if t not in simplicial_models]

model_types_str = ", ".join(type_.value for type_ in ModelType)
