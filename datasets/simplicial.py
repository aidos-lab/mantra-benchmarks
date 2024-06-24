from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from mantra.simplicial import SimplicialDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
import torch
from torch.utils.data import Subset
import numpy as np
from collections import Counter
from typing import List, Dict


def unique_counts(input_list: List[str]):
    return Counter(input_list).keys(), Counter(input_list).values()


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

    def class_imbalance_statistics(self) -> Dict[str, Dict]:
        dataset = SimplicialDataset(root=self.data_dir)

        name_statistics = unique_counts(dataset.name)
        orientability_statistics = unique_counts(dataset.orientable.tolist())

        betti = np.array(dataset.betti_numbers)
        betti_0_statistics = unique_counts(betti[:, 0])
        betti_1_statistics = unique_counts(betti[:, 1])
        betti_2_statistics = unique_counts(betti[:, 2])

        return {
            "name": name_statistics,
            "orientable": orientability_statistics,
            "betti_0": betti_0_statistics,
            "betti_1": betti_1_statistics,
            "betti_2": betti_2_statistics,
        }

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
            random_state=self.seed,
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
