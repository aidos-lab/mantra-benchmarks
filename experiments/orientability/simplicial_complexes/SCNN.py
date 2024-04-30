import math

import lightning as L
import torchvision.transforms as transforms
from torch.utils.data import random_split

from mantra.dataloaders import SimplicialDataLoader
from mantra.simplicial import SimplicialDataset
from mantra.transforms import (
    OrientableToClassSimplicialComplexTransform,
    DimTwoHodgeLaplacianSimplicialComplexTransform,
    DimOneHodgeLaplacianDownSimplicialComplexTransform,
    DimOneHodgeLaplacianUpSimplicialComplexTransform,
    DimZeroHodgeLaplacianSimplicialComplexTransform,
    SimplicialComplexOnesTransform,
)
from mantra.transforms import SimplicialComplexTransform
from models.orientability.simplicial_complexes.SCNNNetworkOrientability import (
    SCNNNetwork,
)


def load_dataset_with_transformations():
    tr = transforms.Compose(
        [
            SimplicialComplexTransform(),
            SimplicialComplexOnesTransform(ones_length=10),
            DimZeroHodgeLaplacianSimplicialComplexTransform(),
            DimOneHodgeLaplacianUpSimplicialComplexTransform(),
            DimOneHodgeLaplacianDownSimplicialComplexTransform(),
            DimTwoHodgeLaplacianSimplicialComplexTransform(),
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
    test_percentage = 0.2
    batch_size = 128
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
    model = SCNNNetwork(
        rank=rank,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        conv_order_down=conv_order_down,
        conv_order_up=conv_order_up,
        n_layers=num_layers,
    )
    test_len = math.floor(len(dataset) * test_percentage)
    train_ds, test_ds = random_split(
        dataset, [len(dataset) - test_len, test_len]
    )
    train_dl = SimplicialDataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    test_dl = SimplicialDataLoader(
        test_ds, batch_size=batch_size, shuffle=False
    )
    # Use CPU acceleration: SCCNN does not support GPU acceleration because it creates matrices not placed in the
    # device of the network.
    trainer = L.Trainer(
        max_epochs=1000, accelerator="cpu", log_every_n_steps=1
    )
    trainer.fit(model, train_dl, test_dl)
