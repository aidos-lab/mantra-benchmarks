import wandb
from typing import Literal

from lightning.pytorch.loggers import WandbLogger


def get_wandb_logger(
    task_name: Literal["orientability", "betti_numbers", "name"],
    save_dir="./lightning_logs",
    model_name: str = None,
    node_features: str = None, 
    run_id: str = None
):
    wandb_logger = WandbLogger(
        project="mantra-dev", entity="er-wnb-1a33y", save_dir=save_dir
    )
    wandb_logger.experiment.config["task"] = task_name
    wandb_logger.experiment.config["run_id"] = run_id
    wandb_logger.experiment.config["node_features"] = node_features
    
    if model_name is not None:
        wandb_logger.experiment.config["model_name"] = model_name
    return wandb_logger
