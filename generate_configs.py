from metrics.tasks import TaskType
from datasets.transforms import TransformType
from models import ModelType, model_cfg_lookup
from experiments.configs import ConfigExperimentRun, TrainerConfig
import yaml
import json
import os
import shutil

tasks = [TaskType.ORIENTABILITY, TaskType.NAME, TaskType.BETTI_NUMBERS]

features = [
    TransformType.degree_transform,
    TransformType.degree_transform_onehot,
    TransformType.random_node_features,
]

models = [
    ModelType.GCN,
    ModelType.GAT,
    ModelType.GCN,
    ModelType.TAG,
    ModelType.TransfConv,
]

node_feature_dict = {
    TransformType.degree_transform: 1,
    TransformType.degree_transform_onehot: 9,
    TransformType.random_node_features: 8,
}

out_channels_dict = {
    TaskType.ORIENTABILITY: 1,
    TaskType.NAME: 5,
    TaskType.BETTI_NUMBERS: 3,
}


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
    for feature in features:
        for task in tasks:
            num_node_features = node_feature_dict[feature]
            out_channels = out_channels_dict[task]
            model_config_cls = model_cfg_lookup[model]
            model_config = model_config_cls(
                out_channels=out_channels, num_node_features=num_node_features
            )
            trainer_config = TrainerConfig(
                accelerator="auto", max_epochs=50, log_every_n_steps=1
            )
            config = ConfigExperimentRun(
                type_model=model,
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
