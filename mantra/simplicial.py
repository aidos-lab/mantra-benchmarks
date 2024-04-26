"""
Simplical dataset, downloads the original data from the website of Frank Lutz
and parses the files into a torch geometric dataset that can be used in 
conjunction to dataloaders. 
"""

from torch_geometric.data import InMemoryDataset, download_url, Data
from mantra.convert import process_manifolds


class SimplicialDataset(InMemoryDataset):
    available_versions = ["1.0.0"]

    @staticmethod
    def _get_raw_dataset_root_link(version: str):
        match version:
            case "1.0.0":
                return "https://www3.math.tu-berlin.de/IfM/Nachrufe/Frank_Lutz/stellar"
            case _:
                raise ValueError(f"Version {version} not available")

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

    @property
    def raw_file_names(self):
        return [
            f"{self.manifold}_manifolds_all.txt",
            f"{self.manifold}_manifolds_all_type.txt",
            f"{self.manifold}_manifolds_all_hom.txt",
        ]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        root_link = self._get_raw_dataset_root_link(self.version)
        for name in self.raw_file_names:
            download_url(
                f"{root_link}/{name}",
                self.raw_dir,
            )

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

        self.save(data_list, self.processed_paths[0])
