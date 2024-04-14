"""
Simplical dataset, downloads the original data from the website of Frank Lutz
and parses the files into a torch geometric dataset that can be used in 
conjunction to dataloaders. 
"""

from torch_geometric.data import InMemoryDataset, download_url, Data
from mantra.convert import process_manifolds


class SimplicialDataset(InMemoryDataset):
    def __init__(
        self, root, transform=None, pre_transform=None, pre_filter=None
    ):
        root += "/simplicial"
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [
            "2_manifolds_all.txt",
            "2_manifolds_all_type.txt",
            "2_manifolds_all_hom.txt",
        ]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        download_url(
            "https://www3.math.tu-berlin.de/IfM/Nachrufe/Frank_Lutz/stellar/2_manifolds_all.txt",
            self.root + "/raw",
        )

        download_url(
            "https://www3.math.tu-berlin.de/IfM/Nachrufe/Frank_Lutz/stellar/2_manifolds_all_type.txt",
            self.root + "/raw",
        )

        download_url(
            "https://www3.math.tu-berlin.de/IfM/Nachrufe/Frank_Lutz/stellar/2_manifolds_all_hom.txt",
            self.root + "/raw",
        )

    def process(self):
        triangulations = process_manifolds(
            f"{self.raw_dir}/2_manifolds_all.txt",
            f"{self.raw_dir}/2_manifolds_all_hom.txt",
            f"{self.raw_dir}/2_manifolds_all_type.txt",
        )

        data_list = [Data(**el) for el in triangulations]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
