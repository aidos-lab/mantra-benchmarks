"""
Simplical dataset, downloads the original data from the website of Frank Lutz
and parses the files into a torch geometric dataset that can be used in 
conjunction to dataloaders. 
"""

import torch
from torch_geometric.data import InMemoryDataset, download_url, Data

from mantra.convert import (
    process_manifolds,
    process_train_test_split_orientability,
)


class SimplicialDataset(InMemoryDataset):
    available_versions = ["1.0.0"]

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        version="1.0.0",
    ):
        assert version in self.available_versions
        self.version = version
        self.manifold = "2"
        root += f"/simplicial_v{version}"
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        self.train_orientability_indices = torch.load(self.processed_paths[1])
        self.test_orientability_indices = torch.load(self.processed_paths[2])

    @property
    def raw_file_names(self):
        return [
            f"{self.manifold}_manifolds_all.txt",
            f"{self.manifold}_manifolds_all_type.txt",
            f"{self.manifold}_manifolds_all_hom.txt",
            f"{self.manifold}_manifolds_all_train_test_split_orientability.txt",
        ]

    @property
    def processed_file_names(self):
        return [
            "data.pt",
            "train_orientability_indices.pt",
            "test_orientability_indices.pt",
        ]

    def _get_download_links(self, version: str):
        match version:
            case "1.0.0":
                root_manifolds = "https://www3.math.tu-berlin.de/IfM/Nachrufe/Frank_Lutz/stellar"
                manifolds_files = [
                    f"{root_manifolds}/{name}"
                    for name in self.raw_file_names
                    if name
                    != f"{self.manifold}_manifolds_all_train_test_split_orientability.txt"
                ]
                orientability_indices_file = [
                    "https://rubenbb.com/assets/MANTRA/v1.0.0/2_manifolds_all_train_test_split_orientability.txt"
                ]
                return manifolds_files + orientability_indices_file
            case _:
                raise ValueError(f"Version {version} not available")

    def download(self):
        download_links = self._get_download_links(self.version)
        for download_link in download_links:
            download_url(download_link, self.raw_dir)

    def process(self):
        triangulations = process_manifolds(
            f"{self.raw_dir}/{self.manifold}_manifolds_all.txt",
            f"{self.raw_dir}/{self.manifold}_manifolds_all_hom.txt",
            f"{self.raw_dir}/{self.manifold}_manifolds_all_type.txt",
        )

        data_list = [Data(**el) for el in triangulations]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        orien_train_indices, orien_test_indices = (
            process_train_test_split_orientability(
                f"{self.raw_dir}/{self.manifold}_manifolds_all_train_test_split_orientability.txt"
            )
        )
        self.save(data_list, self.processed_paths[0])
        torch.save(orien_train_indices, self.processed_paths[1])
        torch.save(orien_test_indices, self.processed_paths[2])
