import os
import typing
from typing import Literal, Tuple, TypeAlias

import torch
from mantra.datasets import ManifoldTriangulations
from sklearn.model_selection import train_test_split

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.transforms import Compose

from metrics.tasks import (
    TaskType,
    class_transforms_lookup_2manifold,
    class_transforms_lookup_3manifold,
    Manifold2Type, Manifold3Type
)

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

Mode: TypeAlias = Literal["train", "test", "val"]

class PairwiseSimplicialDS(InMemoryDataset):
    """
    Wrapper of ManifoldTriangulations to extend it with train/test/val split functionality.

    train/test/val splits depend on the task type due to stratification, i.e. that
    a proper split shall maintain the same class imbalance in all splits.
    Since for every task type the class imbalance differs, stratification can only be done dependent
    on the task type.
    """

    def __init__(
        self,
        root: str,
        split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 0,
        use_stratified: bool = True,
        comparison_pair: Tuple[Manifold3Type, Manifold3Type] | Tuple[Manifold2Type, Manifold2Type] = (Manifold2Type.S_2, Manifold2Type.T_2),
        mode: Mode = "train",
        manifold="2",
        version="latest",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.task_type = TaskType.HOMEO_COMP
        self.manifold = manifold
        self.comparison_pair = comparison_pair
        self.split = mode
        self.split_config = SplitConfig(split, seed, use_stratified)
        self.raw_simplicial_ds = ManifoldTriangulations(
            os.path.join(root, "raw_simplicial"),
            manifold,
            version,
            None,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )
        super().__init__(root, transform=transform)

        self.load(self._get_processed_path(self.comparison_pair[0], self.comparison_pair[1], mode))

    @property
    def raw_file_names(self):
        return []

    def download(self) -> None:
        pass

    def _data_filename(self, m_1: Manifold2Type | Manifold3Type, m_2: Manifold2Type | Manifold3Type, mode: Mode):
        """
            Parameters
            -----------
                m_1: ManifoldType
                    Type of the first manifold the triangulations represent.
                m_2: ManifoldType
                    Type of the second manifold the triangulations represent.
                mode: Mode
                    The mode that the dataset should be split into.
            Returns
            -------
                The name of the data file in .pt format.

        """
        return f"data_{m_1.name.lower()}_{m_2.name.lower()}_{mode}.pt"

    def _get_processed_path(self, m_1: Manifold2Type | Manifold3Type, m_2: Manifold2Type | Manifold3Type, mode: Mode):
        fnames = self.processed_file_names
        idx = 0
        for fname in fnames:
            if fname == self._data_filename(m_1, m_2, mode):
                return self.processed_paths[idx]
            idx += 1
        raise ValueError(
            "Can not find processed data: Unknown config with task type and mode."
        )

    @property
    def processed_file_names(self):
        if self.manifold == "2":
            f_names = [
                self._data_filename(m_1, m_2, mode)
                for m_1 in Manifold2Type 
                for m_2 in Manifold2Type
                for mode in typing.get_args(Mode)
            ]
        else:
            raise NotImplementedError("TODO")
        return f_names

    def process(self):
        print("(preparing train/val/test split)")

        if self.manifold == "3":
            raise NotImplementedError("TODO")

        m_1, m_2 = self.comparison_pair
        indices = range(self.raw_simplicial_ds.len())

        class_transforms_lookup = (
            class_transforms_lookup_3manifold
            if self.manifold == "3"
            else class_transforms_lookup_2manifold
        )

        class_transform = Compose(class_transforms_lookup[self.task_type])

        data_list_preprocessed = [
            class_transform(self.raw_simplicial_ds.get(idx))
            for idx in indices
        ]

        data_list = []
        # NOTE: Makeup pairwise comparisons between the triangulations
        for i, _m1 in enumerate(data_list_preprocessed):
            for j, _m2 in enumerate(data_list_preprocessed):
                # NOTE: Do not add a pair of the same triangulation
                if i == j:
                    continue
                    
                # We are not interested in this pairwise comparison
                if _m1.y not in self.comparison_pair or _m2.y not in self.comparison_pair:
                    continue

                # Create the data object with two triangulations
                data_obj = Data(
                    sc_1 = _m1.sc,
                    sc_2 = _m2.sc,
                    num_nodes_1 = _m1.num_nodes,
                    num_nodes_2 = _m2.num_nodes,
                    y = torch.tensor(_m1.y == _m2.y, dtype=torch.bool)
                )
                data_list.append(data_obj)

        stratified = torch.vstack([data.y for data in data_list])
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
            train_data_list, self._get_processed_path(m_1, m_2, "train")
        )
        self.save(
            val_data_list, self._get_processed_path(m_1, m_2, "val")
        )
        self.save(
            test_data_list, self._get_processed_path(m_1, m_2, "test")
        )
