from metrics.tasks import (
    TaskType,
    class_transforms_lookup_2manifold,
    class_transforms_lookup_3manifold,
)
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, List, Literal, Tuple, Optional
from torch_geometric.transforms import Compose
from mantra.simplicial import SimplicialDataset
from torch_geometric.data import InMemoryDataset
import os


class SplitConfig:

    def __init__(
        self,
        split: Tuple[float, float, float],
        seed: int,
        use_stratified: bool,
    ) -> None:
        self.split = split
        self.seed = seed
        self.use_stratified = use_stratified


Mode = Literal["train", "test", "val"]


class SimplicialDS(InMemoryDataset):
    """
    Wrapper of SimplicialDataset to extend it with train/test/val split functionality.

    train/test/val splits depend on the task type due to stratification, i.e. that
    a proper split shall maintain the same class imbalance in all splits.
    Since for every task type the class imbalance differs, stratification can only be done dependent
    on the task type.
    """

    def __init__(
        self,
        root: str,
        split: Tuple[float, float, float] = [0.8, 0.1, 0.1],
        seed: int = 0,
        use_stratified: bool = True,
        task_type: TaskType = TaskType.ORIENTABILITY,
        mode: Mode = "train",
        manifold="2",
        version="latest",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.manifold = manifold
        self.task_type = task_type
        self.split = mode
        self.split_config = SplitConfig(split, seed, use_stratified)
        self.raw_simplicial_ds = SimplicialDataset(
            os.path.join(root, "raw_simplicial"),
            manifold,
            version,
            None,
            None,
            pre_filter=pre_filter,
        )
        super().__init__(
            root, transform=transform, pre_transform=pre_transform
        )

        self.load(self._get_processed_path(task_type, mode))

    @property
    def raw_file_names(self):
        return []

    def download(self) -> None:
        pass

    def _data_filename(self, task_type: TaskType, mode: Mode):
        return f"data_{task_type.name.lower()}_{mode}.pt"

    def _get_processed_path(self, task_type: TaskType, mode: Mode):
        fnames = self.processed_file_names
        idx = 0
        for fname in fnames:
            if fname == self._data_filename(task_type, mode):
                return self.processed_paths[idx]
            idx += 1
        raise ValueError(
            "Can not find processed data: Unknown config with task type and mode."
        )

    @property
    def processed_file_names(self):
        if self.manifold == "2":
            f_names = [
                self._data_filename(task_type, mode)
                for task_type in TaskType
                for mode in ["train", "test", "val"]
            ]
        else:
            f_names = [
                self._data_filename(task_type, mode)
                for task_type in [
                    TaskType.BETTI_NUMBERS,
                    TaskType.ORIENTABILITY,
                ]
                for mode in ["train", "test", "val"]
            ]
        return f_names

    def process(self):
        print("(preparing train/val/test split)")
        indices = range(self.raw_simplicial_ds.len())

        for task_type in TaskType:

            # no name classification on 3 manifolds
            if self.manifold == "3" and (task_type == TaskType.NAME):
                continue

            # apply class transform
            class_transforms_lookup = (
                class_transforms_lookup_3manifold
                if self.manifold == "3"
                else class_transforms_lookup_2manifold
            )
            class_transform = Compose(class_transforms_lookup[task_type])
            data_list_processed = [
                class_transform(self.raw_simplicial_ds.get(idx))
                for idx in indices
            ]

            # train test split
            stratified = torch.vstack([data.y for data in data_list_processed])
            train_val_indices, test_indices = train_test_split(
                indices,
                test_size=self.split_config.split[2],
                shuffle=True,
                stratify=(
                    stratified.numpy()
                    if self.split_config.use_stratified
                    else None
                ),
                random_state=self.split_config.seed,
            )

            # train val split
            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=self.split_config.split[1]
                / (self.split_config.split[0] + self.split_config.split[1]),
                shuffle=True,
                stratify=(
                    stratified[train_val_indices]
                    if self.split_config.use_stratified
                    else None
                ),
                random_state=self.split_config.seed,
            )

            # save splits
            data_list = [self.raw_simplicial_ds.get(idx) for idx in indices]
            train_data_list = [data_list[idx] for idx in train_indices]
            val_data_list = [data_list[idx] for idx in val_indices]
            test_data_list = [data_list[idx] for idx in test_indices]

            self.save(
                train_data_list, self._get_processed_path(task_type, "train")
            )
            self.save(
                val_data_list, self._get_processed_path(task_type, "val")
            )
            self.save(
                test_data_list, self._get_processed_path(task_type, "test")
            )
