import os
from sympy import deg
import wandb
from mantra.dataset import SimplicialDataModule
from models.GCN import GCN
from models.GAT import GAT
from torch_geometric.transforms import Compose
from lightning.pytorch.loggers import WandbLogger

from models.base import BaseModel
from mantra.transforms import (
    orientability_transforms,
    name_transforms,
    betti_numbers_transforms,
    degree_transform,
    degree_transform_onehot,
    random_node_features,
)

from metrics.accuracies import (
    compute_betti_numbers_accuracies,
    compute_name_accuracies,
    compute_orientability_accuracies,
)
from metrics.loss import (
    name_loss_fn,
    orientability_loss_fn,
    betti_loss_fn,
)
from metrics.metrics import (
    get_betti_numbers_metrics,
    get_name_metrics,
    get_orientability_metrics,
)
from loggers import get_wandb_logger
import lightning as L
from omegaconf import OmegaConf

config = OmegaConf.load("./configs/gcn_orientability_random_config.yaml")

# ===============================================
# Compile all the dictionaries for the
# tasks for hot swaps.
# ===============================================

model_dict = {"GCN": GCN, "GAT": GAT}

transforms_dict = {
    "degree_transform": degree_transform,
    "degree_transform_onehot": degree_transform_onehot,
    "random_node_features": random_node_features,
}


def run_experiment(config, path):
    task_dict = {
        "name": {
            "transforms": Compose(
                transforms_dict[config.data.transforms] + name_transforms
            ),
            "loss_fn": name_loss_fn,
            "metrics": get_name_metrics,
            "accuracies": compute_name_accuracies,
        },
        "orientability": {
            "transforms": Compose(
                transforms_dict[config.data.transforms]
                + orientability_transforms
            ),
            "loss_fn": orientability_loss_fn,
            "metrics": get_orientability_metrics,
            "accuracies": compute_orientability_accuracies,
        },
        "betti_numbers": {
            "transforms": Compose(
                transforms_dict[config.data.transforms]
                + betti_numbers_transforms
            ),
            "loss_fn": betti_loss_fn,
            "metrics": get_betti_numbers_metrics,
            "accuracies": compute_betti_numbers_accuracies,
        },
    }

    print("=====================================")
    print("============CONFIG===================")
    print("=====================================")

    dm = SimplicialDataModule(
        data_dir="./data",
        transform=task_dict[config.task]["transforms"],
        use_stratified=config.data.use_stratified,
        seed=config.data.seed,
    )

    model = model_dict[config.model.model_name](config.model)

    litmodel = BaseModel(
        model,
        *task_dict[config.task]["metrics"](),
        task_dict[config.task]["accuracies"],
        task_dict[config.task]["loss_fn"],
        learning_rate=config.litmodel.learning_rate,
    )

    logger = get_wandb_logger(
        task_name=config.task, model_name=config.model.model_name
    )
    trainer = L.Trainer(
        logger=logger,
        accelerator=config.trainer.accelerator,
        max_epochs=config.trainer.max_epochs,
        log_every_n_steps=config.trainer.log_every_n_steps,
        fast_dev_run=True,
    )

    trainer.fit(litmodel, dm)
    logger.experiment.finish()


def run_test_configs():
    config_dir = "./test_configs"
    files = os.listdir(config_dir)
    for file in files:
        config_file = os.path.join(config_dir, file)
        conf = OmegaConf.load(config_file)
        run_experiment(conf, config_file)


run_test_configs()
# conf = OmegaConf.load("./configs/config.yaml")
# run_experiment(conf, path="")
