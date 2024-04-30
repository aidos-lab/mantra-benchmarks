# TODO
def load_dataset_with_transformations():
    tr = transforms.Compose(
        [
            SimplicialComplexTransform(),
            SimplicialComplexOnesTransform(ones_length=10),
            DimZeroHodgeLaplacianSimplicialComplexTransform(),
            DimOneHodgeLaplacianUpSimplicialComplexTransform(),
            DimOneHodgeLaplacianDownSimplicialComplexTransform(),
            DimTwoHodgeLaplacianSimplicialComplexTransform(),
            OrientableToClassSimplicialComplexTransform(),
        ]
    )
    dataset = SimplicialDataset(root="./data", transform=tr)
    return dataset


def single_experiment_orientability_gcn():
    dataset = load_dataset_with_transformations()
