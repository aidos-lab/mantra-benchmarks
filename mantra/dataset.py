from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from mantra.simplicial import SimplicialDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
import torch
from torch.utils.data import Subset
import numpy as np
from numpy.random import RandomState


class SimplicialDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        transform: Compose | None = None,
        use_stratified: bool = False,
        batch_size: int = 128,
        seed: int = 2024,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.use_stratified = use_stratified
        self.stratified = None
        self.batch_size = batch_size
        self.seed = seed

    def prepare_data(self) -> None:
        SimplicialDataset(root=self.data_dir)

    def setup(self, stage=None):
        simplicial_full = SimplicialDataset(
            root=self.data_dir, transform=self.transform
        )
        if self.use_stratified:
            self.stratified = torch.vstack(
                [data.y for data in simplicial_full]
            )

        indices_dataset = np.arange(len(simplicial_full))

        train_indices, val_indices = train_test_split(
            indices_dataset,
            test_size=0.2,
            shuffle=True,
            stratify=self.stratified,
            # random_state=RandomState(self.seed),
        )
        self.train_ds = Subset(simplicial_full, train_indices)
        self.val_ds = Subset(simplicial_full, val_indices)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)


if __name__ == "__main__":
    dm = SimplicialDataModule()
