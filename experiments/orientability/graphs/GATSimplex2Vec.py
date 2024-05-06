import torch
import torch
import torchvision.transforms as transforms
from torch_geometric.transforms import FaceToEdge

from experiments.experiment_utils import perform_experiment
from experiments.lightning_modules.GraphCommonModuleOrientability import (
    GraphCommonModuleOrientability,
)
from mantra.simplicial import SimplicialDataset
from mantra.transforms import (
    TriangulationToFaceTransform,
    SetNumNodesTransform,
    DegreeTransform,
    OrientableToClassTransform,
    Simplex2VecTransform,
)
from models.graphs.GAT import GATNetwork


class GATSimplexToVecModule(GraphCommonModuleOrientability):
    def __init__(
        self,
        hidden_channels,
        num_node_features,
        out_channels,
        num_heads,
        num_hidden_layers,
        learning_rate=0.0001,
    ):
        base_model = GATNetwork(
            hidden_channels=hidden_channels,
            num_node_features=num_node_features,
            out_channels=out_channels,
            num_heads=num_heads,
            num_hidden_layers=num_hidden_layers,
        )
        super().__init__(base_model=base_model)
        self.learning_rate = learning_rate

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def load_dataset_with_transformations():
    tr = transforms.Compose(
        [
            TriangulationToFaceTransform(),
            SetNumNodesTransform(),
            FaceToEdge(remove_faces=False),
            DegreeTransform(),
            OrientableToClassTransform(),
            Simplex2VecTransform(),
        ]
    )
    dataset = SimplicialDataset(root="./data", transform=tr)
    return dataset


def single_experiment_orientability_gat_simplex2vec():
    # ===============================
    # Training parameters
    # ===============================
    hidden_channels = 64
    num_hidden_layers = 2
    num_heads = 4
    batch_size = 32
    max_epochs = 100
    learning_rate = 0.0001
    num_workers = 0
    # ===============================
    dataset = load_dataset_with_transformations()
    model = GATSimplexToVecModule(
        hidden_channels=hidden_channels,
        num_node_features=dataset.num_node_features,
        out_channels=1,
        num_heads=num_heads,
        num_hidden_layers=num_hidden_layers,
        learning_rate=learning_rate,
    )
    perform_experiment(
        task="orientability",
        model=model,
        model_name="GATSimplex2Vec",
        dataset=dataset,
        batch_size=batch_size,
        max_epochs=max_epochs,
        num_workers=num_workers,
    )
