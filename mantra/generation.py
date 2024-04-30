from sklearn.model_selection import train_test_split

from mantra.convert import process_manifolds
import numpy as np


def generate_random_split(
    all_dataset_triangulations_path,
    all_dataset_homology_path,
    all_dataset_type_path,
    test_size=0.2,
    output_filename="./data/train_test_split_orientability.txt",
):
    processed_manifolds = process_manifolds(
        all_dataset_triangulations_path,
        all_dataset_homology_path,
        all_dataset_type_path,
    )
    indices_dataset = np.arange(len(processed_manifolds))
    orientability_labels = np.array(
        [int(manifold["orientable"]) for manifold in processed_manifolds]
    )
    X_train, X_test = train_test_split(
        indices_dataset,
        test_size=test_size,
        shuffle=True,
        stratify=orientability_labels,
    )
    # Create a txt file with the indices of the train and test set
    with open(output_filename, "w") as f:
        f.write("Train indices: " + " ".join(map(str, X_train)) + "\n")
        f.write("Test indices: " + " ".join(map(str, X_test)) + "\n")


if __name__ == "__main__":
    generate_random_split(
        all_dataset_triangulations_path="../data/simplicial_v1.0.0/raw/2_manifolds_all.txt",
        all_dataset_homology_path="../data/simplicial_v1.0.0/raw/2_manifolds_all_hom.txt",
        all_dataset_type_path="../data/simplicial_v1.0.0/raw/2_manifolds_all_type.txt",
        test_size=0.2,
        output_filename="../data/train_test_split_orientability.txt",
    )
