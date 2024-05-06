from experiments.betti_numbers.graphs.GCN import (
    single_experiment_betti_numbers_gnn,
)
from experiments.orientability.graphs.GATSimplex2Vec import (
    single_experiment_orientability_gat_simplex2vec,
)
from experiments.orientability.graphs.GCN import (
    single_experiment_orientability_gnn,
)
from experiments.orientability.simplicial_complexes.SCNN import (
    single_experiment_orientability_scnn,
)
from experiments.name.graphs.GCN import (
    single_experiment_name_gnn,
)


if __name__ == "__main__":
    # single_experiment_orientability_gnn()
    # single_experiment_orientability_scnn()
    # single_experiment_orientability_gat_simplex2vec()
    # single_experiment_betti_numbers_gnn()
    single_experiment_name_gnn()
