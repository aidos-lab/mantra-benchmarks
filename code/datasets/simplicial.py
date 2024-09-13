from typing import Callable
from lightning import LightningDataModule
from torch_geometric.loader import DataLoader as DataLoaderGeometric
from torch_geometric.transforms import Compose
from collections import Counter
from typing import List
from .simplicial_ds import SimplicialDS
from metrics.tasks import TaskType
from datasets.dataset_types import DatasetType, filter_nameless
import os


def unique_counts(input_list: List[str]) -> Counter:
    return Counter(input_list)


class SimplicialDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        task_type: TaskType = TaskType.ORIENTABILITY,
        transform: Compose | None = None,
        use_stratified: bool = False,
        batch_size: int = 128,
        seed: int = 2024,
        dataloader_builder: Callable = DataLoaderGeometric,
        ds_type: DatasetType = DatasetType.FULL_2D,
    ):
        super().__init__()
        self.ds_type = ds_type
        self.data_dir = data_dir
        self.transform = transform
        self.task_type = task_type
        self.use_stratified = use_stratified
        self.stratified = None
        self.batch_size = batch_size
        self.seed = seed
        self.dataloader_builder = dataloader_builder
        self.split = [0.6, 0.2, 0.2]

    def get_ds(self, mode: str = "train") -> SimplicialDS:
        if self.ds_type == DatasetType.FULL_2D:
            manifold = "2"
            pre_filter = None
        elif self.ds_type == DatasetType.FULL_3D:
            manifold = "3"
            pre_filter = None
        elif self.ds_type == DatasetType.NO_NAMELESS_2D:
            manifold = "2"
            pre_filter = filter_nameless
        else:
            raise ValueError(f"Unknown dataset type {self.ds_type}")

        return SimplicialDS(
            root=os.path.join(self.data_dir, self.ds_type.name.lower()),
            manifold=manifold,
            split=self.split,
            seed=self.seed,
            mode=mode,
            use_stratified=self.use_stratified,
            task_type=self.task_type,
            transform=self.transform,
            pre_filter=pre_filter,
        )

    def prepare_data(self) -> None:
        self.get_ds()

    def class_imbalance_statistics(self) -> Counter:
        dataset = self.get_ds()

        statistics = None
        if self.task_type == TaskType.NAME:
            statistics = unique_counts(dataset.name)
        elif self.task_type == TaskType.ORIENTABILITY:
            statistics = unique_counts(dataset.orientable.tolist())
        else:
            raise NotImplementedError()
        return statistics

    def setup(self, stage=None):
        self.train_ds = self.get_ds("train")
        self.val_ds = self.get_ds("val")
        self.test_ds = self.get_ds("test")

    def train_dataloader(self):
        return self.dataloader_builder(
            self.train_ds, batch_size=self.batch_size
        )

    def val_dataloader(self):
        return self.dataloader_builder(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return self.dataloader_builder(
            self.test_ds, batch_size=self.batch_size
        )


if __name__ == "__main__":
    dm = SimplicialDataModule()
