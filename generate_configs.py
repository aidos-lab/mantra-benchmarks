from omegaconf import OmegaConf

from models.GCN import GCN, GCNConfig
from models.GAT import GAT, GATConfig
from models.MLP import MLP, MLPConfig 
from models.TransfConv import TransfConv, TransfConvConfig
from models.TAG import TAG, TAGConfig

tasks = ["orientability", "name", "betti_numbers"]

features = [
    "degree_transform",
    "degree_transform_onehot",
    "random_node_features",
]

models = ["GCN", "GAT", "MLP", "TAG", "TransfConv"]
# models = ["TransfConv"]

models_dict = {"GCN": GCNConfig, "GAT": GATConfig, "MLP": MLPConfig,"TAG":TAGConfig, "TransfConv":TransfConvConfig}


node_feature_dict = {
    "degree_transform": 1,
    "degree_transform_onehot": 9,
    "random_node_features": 8,
}

out_channels_dict = {
    "orientability": 1,
    "name": 5,
    "betti_numbers": 3,
}

for model in models:
    for feature in features:
        for task in tasks:
            num_node_features = node_feature_dict[feature]
            out_channels = out_channels_dict[task]

            modelconfig = models_dict[model](
                num_node_features=num_node_features, out_channels=out_channels
            ).__dict__
            trainer = {
                "accelerator": "auto",
                "max_epochs": 50,
                "log_every_n_steps": 1,
            }
            litmodel = {"learning_rate": 0.01}

            data = {
                "transforms": feature,
                "use_stratified": False if task == "betti_numbers" else True,
                "seed": 1234,
            }

            config = {
                "task": task,
                "model": modelconfig,
                "trainer": trainer,
                "litmodel": litmodel,
                "data": data,
            }

            conf = OmegaConf.create(config)
            OmegaConf.save(
                conf, f"./configs/{model.lower()}_{task}_{feature}.yaml"
            )
