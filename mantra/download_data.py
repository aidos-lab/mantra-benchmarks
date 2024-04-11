"""
A download script for development purposes, can be removed later.
"""

import requests


def download_data():

    response = requests.get(
        "https://www3.math.tu-berlin.de/IfM/Nachrufe/Frank_Lutz/stellar/2_manifolds_all.txt"
    )
    with open("./raw_data/2_manifolds_all.txt", "w") as f:
        f.write(response.text)

    response = requests.get(
        "https://www3.math.tu-berlin.de/IfM/Nachrufe/Frank_Lutz/stellar/2_manifolds_all_type.txt"
    )
    with open("./raw_data/2_manifolds_type.txt", "w") as f:
        f.write(response.text)

    response = requests.get(
        "https://www3.math.tu-berlin.de/IfM/Nachrufe/Frank_Lutz/stellar/2_manifolds_all_hom.txt"
    )
    with open("./raw_data/2_manifolds_hom.txt", "w") as f:
        f.write(response.text)
