from enum import Enum


class ModelType(Enum):
    SAN = "san"
    SCCN = "sccn"
    SCCNN = "sccnn"
    SCN = "scn"
    GAT = "gat"
    GCN = "gcn"
    MLP = "mlp"
    CELL_TRANSF = "celltrans"
    TAG = "tag"
    TransfConv = "transfconv"


simplicial_models = [
    ModelType.SAN,
    ModelType.SCCN,
    ModelType.SCCNN,
    ModelType.SCN,
    ModelType.CELL_TRANSF
]
graphbased_models = [t for t in ModelType if t not in simplicial_models]

model_types_str = ", ".join(type_.value for type_ in ModelType)
