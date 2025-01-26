import os
import sys

sys.path.append(os.curdir)
from metrics.tasks import TaskType
from datasets.transforms import TransformType
from datasets.dataset_types import DatasetType
from models import ModelType, model_cfg_lookup
from experiments.utils.configs import ConfigExperimentRun, TrainerConfig
import yaml
import json
import os
import shutil
import argparse
from typing import List, Dict

# ARGS ------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Argument parser for experiment configurations."
)
parser.add_argument(
    "--max_epochs",
    type=int,
    default=10,
    help="Maximum number of epochs.",
)

parser.add_argument(
    "--config_dir",
    type=str,
    default="./configs",
    help="Directory where config files shall be stored",
)

parser.add_argument(
    "--lr",
    type=float,
    default=0.01,
    help="Maximum number of epochs.",
)

parser.add_argument(
    "--use_imbalance_weighting",
    action="store_true",
    help="Whether to weight loss terms with imbalance weights.",
)

parser.add_argument(
    "--three_manifold_only",
    action="store_true",
    help="Whether to use only the 3-manifold dataset only.",
)

parser.add_argument(
    "--random_transform_only",
    action="store_true",
    help="Whether to use random transform only.",
)

parser.add_argument(
    "--degree_transform_only",
    action="store_true",
    help="Whether to use degree_transform_only only.",
)

args = parser.parse_args()
max_epochs: int = args.max_epochs
lr: float = args.lr
config_dir: str = args.config_dir
use_imbalance_weights: bool = args.use_imbalance_weighting
three_manifold_only: bool = args.three_manifold_only
random_transform_only: bool = args.random_transform_only
degree_transform_only: bool = args.degree_transform_only
# -----------------------------------------------------------------------------

# CONFIGS ---------------------------------------------------------------------

# DS TYPE ###
dataset_types = [
    DatasetType.FULL_2D,
    DatasetType.FULL_3D,
    DatasetType.NO_NAMELESS_2D,
]
# ###########

# TASKS #####
tasks_mantra2 = [TaskType.ORIENTABILITY, TaskType.NAME, TaskType.BETTI_NUMBERS]
tasks_mantra3 = [TaskType.BETTI_NUMBERS, TaskType.ORIENTABILITY]
# ###########

# TRANSFORMS
graph_features_degree = [
    TransformType.degree_transform,
    TransformType.degree_transform_onehot,
]

graph_features_random = [
    TransformType.random_node_features,
]

if degree_transform_only and random_transform_only:
    raise ValueError()

graph_features = graph_features_random + graph_features_degree

simplicial_features_2d = [
    TransformType.degree_transform_sc_2d,
    TransformType.random_simplices_features_2d,
]

simplicial_features_3d = [
    TransformType.degree_transform_sc_3d,
    TransformType.random_simplices_features_3d,
]

if degree_transform_only:
    graph_features = graph_features_degree
    simplicial_features_2d = [TransformType.degree_transform_sc_2d]
    simplicial_features_3d = [TransformType.degree_transform_sc_3d]

if random_transform_only:
    graph_features = graph_features_random
    simplicial_features_2d = [TransformType.random_simplices_features_2d]
    simplicial_features_3d = [TransformType.random_simplices_features_3d]


# ###########

# MODELS ####
graph_models = {
    ModelType.GCN,
    ModelType.GAT,
    ModelType.MLP,
    ModelType.TAG,
    ModelType.TransfConv,
    ModelType.DECT,
}
simplicial_models = {
    ModelType.SAN,
    ModelType.SCCN,
    ModelType.SCCNN,
    ModelType.SCN,
    ModelType.CELL_TRANSF,
    ModelType.CELL_MP,
}
models = list(graph_models) + list(simplicial_models)
# ###########

# MISC ######


def get_feature_dim(tr_type: TransformType, model_type: ModelType):
    feature_dim_dict = {
        TransformType.degree_transform: 1,
        TransformType.degree_transform_onehot: 10,
        TransformType.random_node_features: 8,
        TransformType.degree_transform_sc_2d: [1, 2, 1],
        TransformType.random_simplices_features_2d: [8, 8, 8],
        TransformType.degree_transform_sc_3d: [1, 2, 2, 1],
        TransformType.random_simplices_features_3d: [8, 8, 8, 8],
    }

    feature_dim_dict_cell_mp = {
        TransformType.degree_transform_sc_3d: 3,
        TransformType.random_simplices_features_3d: 8,
        TransformType.degree_transform_sc_2d: 3,
        TransformType.random_simplices_features_2d: 8,
    }
    if model_type == ModelType.CELL_MP:
        return feature_dim_dict_cell_mp[tr_type]
    else:
        return feature_dim_dict[tr_type]


out_channels_dict_mantra2_full = {
    TaskType.ORIENTABILITY: 1,
    TaskType.NAME: 5,
    TaskType.BETTI_NUMBERS: 3,
}
out_channels_dict_mantra2_no_nameless = {
    TaskType.ORIENTABILITY: 1,
    TaskType.NAME: 4,
    TaskType.BETTI_NUMBERS: 3,
}
out_channels_dict_mantra3 = {
    TaskType.ORIENTABILITY: 1,
    TaskType.BETTI_NUMBERS: 4,
}


# ###########

# -----------------------------------------------------------------------------


# UTILS -----------------------------------------------------------------------
def get_feature_types(model: ModelType, ds_type: DatasetType):
    if model in graph_models:
        return graph_features
    else:
        if ds_type in [DatasetType.FULL_2D, DatasetType.NO_NAMELESS_2D]:
            return simplicial_features_2d
        return simplicial_features_3d


def get_model_config(
    model: ModelType, out_channels: int, dim_features: int | tuple[int]
):
    model_config_cls = model_cfg_lookup[model]
    if model in graph_models:
        model_config = model_config_cls(
            out_channels=out_channels, num_node_features=dim_features
        )
    elif model == ModelType.CELL_TRANSF:
        model_config = model_config_cls(
            input_sizes={
                i: dim_feat for i, dim_feat in enumerate(dim_features)
            },
            positional_encodings_lengths={
                i: 8 for i in range(len(dim_features))
            },
            out_size=out_channels,
        )
    elif model == ModelType.CELL_MP:
        model_config = model_config_cls(
            num_input_features=dim_features,
            num_classes=out_channels,
        )
    else:
        model_config = model_config_cls(
            out_channels=out_channels, in_channels=tuple(dim_features)
        )
    return model_config


def manage_directory(path: str):
    """
    Removes directory if exists and creates and empty directory
    """
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    os.makedirs(path)


def get_tasks(ds_type: DatasetType) -> List[TaskType]:
    tasks = (
        tasks_mantra2.copy()
        if (
            ds_type == DatasetType.FULL_2D
            or ds_type == DatasetType.NO_NAMELESS_2D
        )
        else tasks_mantra3.copy()
    )
    return tasks


def get_out_channels_dict(ds_type: DatasetType) -> Dict[TaskType, int]:
    if ds_type == DatasetType.FULL_2D:
        return out_channels_dict_mantra2_full.copy()
    elif ds_type == DatasetType.FULL_3D:
        return out_channels_dict_mantra3.copy()
    elif ds_type == DatasetType.NO_NAMELESS_2D:
        return out_channels_dict_mantra2_no_nameless.copy()
    else:
        raise ValueError("Unknown dataset type")


# -----------------------------------------------------------------------------

# GENERATE --------------------------------------------------------------------
manage_directory(config_dir)

for ds_type in dataset_types:

    if three_manifold_only and ds_type != DatasetType.FULL_3D:
        continue

    tasks = get_tasks(ds_type)
    for model in models:
        features = get_feature_types(model, ds_type)
        for feature in features:
            for task in tasks:
                dim_features = get_feature_dim(feature, model)

                out_channels = get_out_channels_dict(ds_type)[task]
                model_config = get_model_config(
                    model, out_channels, dim_features
                )
                model_config_cls = model_cfg_lookup[model]
                trainer_config = TrainerConfig(
                    accelerator="auto",
                    max_epochs=max_epochs,
                    log_every_n_steps=1,
                )
                config = ConfigExperimentRun(
                    ds_type=ds_type,
                    task_type=task,
                    seed=1234,
                    transforms=feature,
                    use_stratified=(
                        False if task == TaskType.BETTI_NUMBERS else True
                    ),
                    use_imbalance_weighting=use_imbalance_weights,
                    learning_rate=lr,
                    trainer_config=trainer_config,
                    conf_model=model_config,
                )

                json_string = config.model_dump_json()

                python_dict = json.loads(json_string)
                yaml_string = yaml.dump(python_dict)
                config_identifier = config.get_identifier()
                yaml_file_path = os.path.join(
                    config_dir,
                    f"{config_identifier}.yaml",
                )
                with open(yaml_file_path, "w") as file:
                    file.write(yaml_string)
# -----------------------------------------------------------------------------
