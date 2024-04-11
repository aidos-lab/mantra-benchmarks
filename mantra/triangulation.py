from torch_geometric.data import InMemoryDataset, download_url, Data
import requests
from convert import process_manifold


class SimplicialDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['2_manifolds_all.txt','2_manifolds_all_type.txt','2_manifolds_all_hom.txt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        response = requests.get("https://www3.math.tu-berlin.de/IfM/Nachrufe/Frank_Lutz/stellar/2_manifolds_all.txt")
        with open("./raw/2_manifolds_all.txt","w") as f:
            f.write(response.text)

        response = requests.get("https://www3.math.tu-berlin.de/IfM/Nachrufe/Frank_Lutz/stellar/2_manifolds_all_type.txt")
        with open("./raw/2_manifolds_all_type.txt","w") as f:
            f.write(response.text)

        response = requests.get("https://www3.math.tu-berlin.de/IfM/Nachrufe/Frank_Lutz/stellar/2_manifolds_all_hom.txt")
        with open("./raw/2_manifolds_all_hom.txt","w") as f:
            f.write(response.text)


    def process(self):
        triangulations = process_manifold("./raw/2_manifolds_all.txt","./raw/2_manifolds_all_hom.txt","./raw/2_manifolds_all_type.txt")
        data_list = [Data(**el) for el in triangulations]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])


