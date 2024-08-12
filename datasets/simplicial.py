from typing import Callable

from lightning import LightningDataModule
from torch_geometric.loader import DataLoader as DataLoaderGeometric
from torch_geometric.transforms import Compose
from collections import Counter
from typing import List, Dict
from .simplicial_ds import SimplicialDS
from metrics.tasks import TaskType


def unique_counts(input_list: List[str]):
    return Counter(input_list).keys(), Counter(input_list).values()


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
    ):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.task_type = task_type
        self.use_stratified = use_stratified
        self.stratified = None
        self.batch_size = batch_size
        self.seed = seed
        self.dataloader_builder = dataloader_builder

    def prepare_data(self) -> None:
        SimplicialDS(root=self.data_dir)

    def class_imbalance_statistics(self) -> Dict | None:
        dataset = SimplicialDS(root=self.data_dir, task_type=self.task_type)

        statistics = None
        if self.task_type == TaskType.NAME:
            statistics = unique_counts(dataset.name)
        elif self.task_type == TaskType.ORIENTABILITY:
            statistics = unique_counts(dataset.orientable.tolist())
        else:
            raise NotImplementedError()

        return statistics

    def setup(self, stage=None):
        get_ds = lambda mode: SimplicialDS(
            root=self.data_dir,
            split=[0.7, 0.15, 0.15],
            seed=self.seed,
            use_stratified=self.use_stratified,
            task_type=self.task_type,
            mode=mode,
            transform=self.transform,
        )

        self.train_ds = get_ds("train")
        self.val_ds = get_ds("val")
        self.test_ds = get_ds("test")

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
