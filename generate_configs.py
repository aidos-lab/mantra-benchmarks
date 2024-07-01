from metrics.tasks import TaskType
from datasets.transforms import TransformType
from models import ModelType, model_cfg_lookup
from experiments.configs import ConfigExperimentRun, TrainerConfig
import yaml
import json
import os
import shutil

tasks = [TaskType.ORIENTABILITY, TaskType.NAME, TaskType.BETTI_NUMBERS]

graph_features = [
    TransformType.degree_transform,
    TransformType.degree_transform_onehot,
    TransformType.random_node_features,
]

simplicial_features = [
    TransformType.degree_transform_sc,
    TransformType.random_simplices_features,
]

graph_models = {
    ModelType.GCN,
    ModelType.GAT,
    ModelType.MLP,
    ModelType.TAG,
    ModelType.TransfConv,
}

simplicial_models = {
    ModelType.SAN,
    ModelType.SCCN,
    ModelType.SCCNN,
    ModelType.SCN,
}

models = list(graph_models) + list(simplicial_models)

feature_dim_dict = {
    TransformType.degree_transform: 1,
    TransformType.degree_transform_onehot: 10,
    TransformType.random_node_features: 8,
    TransformType.degree_transform_sc: [1, 2, 1],
    TransformType.random_simplices_features: [8, 8, 8],
}

out_channels_dict = {
    TaskType.ORIENTABILITY: 1,
    TaskType.NAME: 5,
    TaskType.BETTI_NUMBERS: 3,
}


def get_feature_types(model: ModelType):
    if model in graph_models:
        return graph_features
    else:
        return simplicial_features


def get_model_config(
    model: ModelType, out_channels: int, dim_features: int | tuple[int]
):
    model_config_cls = model_cfg_lookup[model]
    if model in graph_models:
        model_config = model_config_cls(
            out_channels=out_channels, num_node_features=dim_features
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


manage_directory("./configs")

for model in models:
    features = get_feature_types(model)
    for feature in features:
        for task in tasks:
            dim_features = feature_dim_dict[feature]
            out_channels = out_channels_dict[task]
            model_config = get_model_config(model, out_channels, dim_features)
            model_config_cls = model_cfg_lookup[model]
            trainer_config = TrainerConfig(
                accelerator="auto", max_epochs=2, log_every_n_steps=1
            )
            config = ConfigExperimentRun(
                task_type=task,
                seed=1234,
                transforms=feature,
                use_stratified=(
                    False if task == TaskType.BETTI_NUMBERS else True
                ),
                learning_rate=0.01,
                trainer_config=trainer_config,
                conf_model=model_config,
            )

            json_string = config.model_dump_json()

            python_dict = json.loads(json_string)
            yaml_string = yaml.dump(python_dict)
            yaml_file_path = f"./configs/{model.name.lower()}_{task.name.lower()}_{feature.name.lower()}.yaml"
            with open(yaml_file_path, "w") as file:
                file.write(yaml_string)
