import torch
from torch import nn
from torch_geometric.transforms import OneHotDegree, FaceToEdge
from torchvision import transforms

from experiments.experiment_utils import perform_experiment
from experiments.lightning_modules.BaseModelClassification import (
    BaseClassificationModule,
)
from mantra.simplicial import SimplicialDataset
from mantra.transforms import (
    DegreeTransform,
    TriangulationToFaceTransform,
    NameToClassTransform,
)
from models.others.MLPConstantShape import MLPConstantShape


class MLPModule(BaseClassificationModule):
    def __init__(
        self,
        num_input_neurons,
        num_hidden_neurons,
        num_hidden_layers,
        num_out_neurons,
        learning_rate,
    ):
        super().__init__(task="name")
        self.base_model = MLPConstantShape(
            num_input_neurons=num_input_neurons,
            num_hidden_neurons=num_hidden_neurons,
            num_hidden_layers=num_hidden_layers,
            num_out_neurons=num_out_neurons,
        )
        self.learning_rate = learning_rate

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.base_model.parameters(), lr=self.learning_rate
        )
        return optimizer

    def forward(self, x, batch):
        x = self.base_model(x, batch)
        return x

    def general_step(self, batch, batch_idx, step: str):
        x_hat = self(batch.x, batch.batch)
        y = batch.y
        batch_len = len(y)
        loss = nn.functional.cross_entropy(x_hat, y)
        self.log(
            f"{step}_loss",
            loss,
            prog_bar=True,
            batch_size=batch_len,
            on_step=False,
            on_epoch=True,
        )
        self.log_accuracies(x_hat, y, batch_len, step)
        return loss


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


def single_experiment_name_mlp_constant_shape():
    # ===============================
    # Training parameters
    # ===============================
    num_hidden_neurons = 64
    num_hidden_layers = 3
    batch_size = 32
    learning_rate = 0.1
    num_workers = 0
    max_epochs = 100
    # ===============================
    dataset = load_dataset_with_transformations()
    model = MLPModule(
        num_input_neurons=dataset.num_features,
        num_hidden_neurons=num_hidden_neurons,
        num_hidden_layers=num_hidden_layers,
        num_out_neurons=5,  # Five different name classes,
        learning_rate=learning_rate,
    )
    perform_experiment(
        task="name",
        model=model,
        model_name="MLPConstantShape",
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        max_epochs=max_epochs,
    )
