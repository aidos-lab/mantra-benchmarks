from typing import Literal

import torch
import torchvision.transforms as transforms
from torch import nn

from experiments.experiment_utils import perform_experiment
from experiments.lightning_modules.BaseModelClassification import (
    BaseClassificationModule,
)
from mantra.dataloaders import SimplicialDataLoader
from mantra.simplicial import SimplicialDataset
from mantra.transforms import (
    OrientableToClassSimplicialComplexTransform,
    SimplicialComplexOnesTransform,
    SCNNNeighborhoodMatricesTransform,
)
from mantra.transforms import SimplicialComplexTransform
from mantra.utils import transfer_simplicial_complex_batch_to_device
from models.simplicial_complexes.SCNN import SCNNNetwork


class SCNNNModule(BaseClassificationModule):
    def __init__(
        self,
        rank,
        in_channels,
        hidden_channels,
        out_channels,
        conv_order_down,
        conv_order_up,
        n_layers=3,
        learning_rate=0.01,
    ):
        super().__init__(task="orientability")
        self.rank = rank
        self.learning_rate = learning_rate
        self.base_model = SCNNNetwork(
            rank=rank,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            conv_order_down=conv_order_down,
            conv_order_up=conv_order_up,
            n_layers=n_layers,
        )

    def forward(self, x, laplacian_down, laplacian_up, signal_belongings):
        x = self.base_model(x, laplacian_down, laplacian_up, signal_belongings)
        return x

    def general_step(
        self, batch, batch_idx, step: Literal["train", "test", "validation"]
    ):
        s_complexes, signal_belongings, batch_len = batch
        x = s_complexes.signals[self.rank]
        if self.rank == 0:
            laplacian_down = None
            laplacian_up = s_complexes.neighborhood_matrices[f"0_laplacian"]
        elif self.rank == 1:
            laplacian_down = s_complexes.neighborhood_matrices[
                f"1_laplacian_down"
            ]
            laplacian_up = s_complexes.neighborhood_matrices[f"1_laplacian_up"]
        elif self.rank == 2:
            laplacian_down = s_complexes.neighborhood_matrices[f"2_laplacian"]
            laplacian_up = None
        else:
            raise ValueError("rank must be 0, 1 or 2.")
        y = s_complexes.other_features["y"].float()
        signal_belongings = signal_belongings[self.rank]
        x_hat = self(x, laplacian_down, laplacian_up, signal_belongings)
        # Squeeze x_hat to match the shape of y
        x_hat = x_hat.squeeze()
        loss = nn.functional.binary_cross_entropy_with_logits(x_hat, y)
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
            SCNNNeighborhoodMatricesTransform(),
            OrientableToClassSimplicialComplexTransform(),
        ]
    )
    dataset = SimplicialDataset(root="./data", transform=tr)
    return dataset


def single_experiment_orientability_scnn():
    dataset = load_dataset_with_transformations()
    # ===============================
    # Training parameters
    # ===============================
    rank = 1  # We work with edge features
    conv_order_down = 2  # TODO: No idea of what this parameter does
    conv_order_up = 2  # TODO: No idea of what this parameter does
    hidden_channels = 20
    out_channels = 1  # num classes
    num_layers = 5
    batch_size = 128
    max_epochs = 100
    learning_rate = 0.01
    num_workers = 0
    # ===============================
    # Checks and dependent parameters
    # ===============================
    # Check the rank has an appropriate value.
    assert 0 <= rank <= 2, "rank must be 0, 1 or 2."
    # select the simplex level
    if rank == 0:
        conv_order_down = 0
    # configure parameters
    in_channels = dataset[0].x[rank].shape[1]
    # ===============================
    # Model and dataloader creation
    # ===============================
    model = SCNNNModule(
        rank=rank,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        conv_order_down=conv_order_down,
        conv_order_up=conv_order_up,
        n_layers=num_layers,
        learning_rate=learning_rate,
    )
    perform_experiment(
        task="orientability",
        model=model,
        model_name="SCNN",
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        max_epochs=max_epochs,
        data_loader_class=SimplicialDataLoader,
        accelerator="cpu",
    )
