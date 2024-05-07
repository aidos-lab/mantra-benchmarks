import torch
from torch import nn
from torchvision import transforms

from experiments.experiment_utils import perform_experiment
from experiments.lightning_modules.BaseModelClassification import (
    BaseClassificationModule,
)
from mantra.dataloaders import SimplicialDataLoader
from mantra.simplicial import SimplicialDataset
from mantra.transforms import (
    SimplicialComplexTransform,
    SimplicialComplexOnesTransform,
    SCConvNeighborhoodMatricesTransform,
    NameToClassSimplicialComplexTransform,
)
from mantra.utils import transfer_simplicial_complex_batch_to_device
from models.simplicial_complexes.SCConv import SCConvNetwork


class SCConvModule(BaseClassificationModule):
    def __init__(
        self, in_channels, out_channels, n_layers=1, learning_rate=0.01
    ):
        # in_channels = hidden_channels
        super().__init__(task="name")
        self.base_model = SCConvNetwork(
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=n_layers,
        )
        self.learning_rate = learning_rate

    def forward(
        self,
        x_0,
        x_1,
        x_2,
        incidence_1,
        incidence_2,
        incidence_1_transpose,
        incidence_2_transpose,
        adjacency_up_0_norm,
        adjacency_up_1_norm,
        adjacency_down_1_norm,
        adjacency_down_2_norm,
        signal_belongings,
    ):
        x = self.base_model(
            x_0,
            x_1,
            x_2,
            incidence_1,
            incidence_2,
            incidence_1_transpose,
            incidence_2_transpose,
            adjacency_up_0_norm,
            adjacency_up_1_norm,
            adjacency_down_1_norm,
            adjacency_down_2_norm,
            signal_belongings,
        )
        return x

    def general_step(self, batch, batch_idx, step: str):
        s_complexes, signal_belongings, batch_len = batch
        x_0 = s_complexes.signals[0]
        x_1 = s_complexes.signals[1]
        x_2 = s_complexes.signals[2]
        incidence_1_transposed_norm = s_complexes.neighborhood_matrices[
            "1_boundary_transpose_norm"
        ]
        incidence_1_norm = s_complexes.neighborhood_matrices["1_boundary_norm"]
        incidence_2_transposed_norm = s_complexes.neighborhood_matrices[
            "2_boundary_transpose_norm"
        ]
        incidence_2_norm = s_complexes.neighborhood_matrices["2_boundary_norm"]
        adjacency_up_0_norm = s_complexes.neighborhood_matrices[
            "0_laplacian_up_norm"
        ]
        adjacency_up_1_norm = s_complexes.neighborhood_matrices[
            "1_laplacian_up_norm"
        ]
        adjacency_down_1_norm = s_complexes.neighborhood_matrices[
            "1_laplacian_down_norm"
        ]
        adjacency_down_2_norm = s_complexes.neighborhood_matrices[
            "2_laplacian_down_norm"
        ]
        x_hat = self(
            x_0,
            x_1,
            x_2,
            incidence_1_norm,
            incidence_2_norm,
            incidence_1_transposed_norm,
            incidence_2_transposed_norm,
            adjacency_up_0_norm,
            adjacency_up_1_norm,
            adjacency_down_1_norm,
            adjacency_down_2_norm,
            signal_belongings,
        )
        y = s_complexes.other_features["y"]
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

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return transfer_simplicial_complex_batch_to_device(
            batch, device, dataloader_idx
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.base_model.parameters(), lr=self.learning_rate
        )
        return optimizer


def load_dataset_with_transformations():
    tr = transforms.Compose(
        [
            SimplicialComplexTransform(),
            SimplicialComplexOnesTransform(ones_length=10),
            SCConvNeighborhoodMatricesTransform(),
            NameToClassSimplicialComplexTransform(),
        ]
    )
    dataset = SimplicialDataset(root="./data", transform=tr)
    return dataset


def single_experiment_name_scconv():
    dataset = load_dataset_with_transformations()
    # ===============================
    # Training parameters
    # ===============================
    num_layers = 5
    batch_size = 128
    max_epochs = 100
    learning_rate = 0.01
    num_workers = 0

    # configure parameters
    in_channels = dataset[0].x[0].shape[1]
    # ===============================
    # Model and dataloader creation
    # ===============================
    model = SCConvModule(
        in_channels=in_channels,
        out_channels=5,  # Five different name classes
        n_layers=num_layers,
        learning_rate=learning_rate,
    )
    perform_experiment(
        task="name",
        model=model,
        model_name="SCConv",
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        max_epochs=max_epochs,
        data_loader_class=SimplicialDataLoader,
        accelerator="cpu",
    )
