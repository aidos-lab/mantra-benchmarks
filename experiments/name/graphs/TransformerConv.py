import torch
from torch_geometric.transforms import FaceToEdge, OneHotDegree
from torchvision import transforms

from experiments.experiment_utils import perform_experiment
from experiments.lightning_modules.GraphCommonModuleName import (
    GraphCommonModuleName,
)
from mantra.simplicial import SimplicialDataset
from mantra.transforms import (
    TriangulationToFaceTransform,
    DegreeTransform,
    NameToClassTransform,
)
from models.TransformerConv import TransformerConvNetwork


class TransformerConvModule(GraphCommonModuleName):
    def __init__(
        self,
        hidden_channels,
        num_node_features,
        out_channels,
        num_hidden_layers,
        learning_rate=0.01,
    ):
        base_model = TransformerConvNetwork(
            hidden_channels=hidden_channels,
            num_node_features=num_node_features,
            out_channels=out_channels,
            num_hidden_layers=num_hidden_layers,
        )
        super().__init__(base_model=base_model)
        self.learning_rate = learning_rate

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.base_model.parameters(), lr=self.learning_rate
        )
        return optimizer


def load_dataset_with_transformations():
    tr = transforms.Compose(
        [
            TriangulationToFaceTransform(),
            FaceToEdge(remove_faces=False),
            DegreeTransform(),
            OneHotDegree(max_degree=8, cat=False),
            NameToClassTransform(),
        ]
    )
    dataset = SimplicialDataset(root="./data", transform=tr)
    return dataset


def single_experiment_name_transformer_conv():
    # ===============================
    # Training parameters
    # ===============================
    hidden_channels = 64
    num_hidden_layers = 2
    batch_size = 32
    max_epochs = 100
    learning_rate = 0.1
    num_workers = 0
    # ===============================
    dataset = load_dataset_with_transformations()
    model = TransformerConvModule(
        hidden_channels=hidden_channels,
        num_node_features=dataset.num_node_features,
        out_channels=5,  # Five different name classes
        num_hidden_layers=num_hidden_layers,
        learning_rate=learning_rate,
    )
    perform_experiment(
        task="name",
        model=model,
        model_name="TransformerConv",
        dataset=dataset,
        batch_size=batch_size,
        max_epochs=max_epochs,
        num_workers=num_workers,
    )
