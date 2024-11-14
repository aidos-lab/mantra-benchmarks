"""
Logging via wandb.
"""

from lightning.pytorch.loggers import WandbLogger

from metrics.tasks import TaskType


def get_wandb_logger(
        task_name: TaskType,
        save_dir="./lightning_logs",
        model_name: str = None,
        node_features: str = None,
        run_id: str = None,
        project_id: str = "mantra-dev",
):
    wandb_logger = WandbLogger(project=project_id, save_dir=save_dir)
    return wandb_logger


def update_wandb_logger(
        wandb_logger,
        task_name: TaskType,
        save_dir="./lightning_logs",
        model_name: str = None,
        node_features: str = None,
        run_id: str = None,
        project_id: str = "mantra-dev",
):
    wandb_logger.experiment.config["task"] = task_name.lower()
    wandb_logger.experiment.config["run_id"] = run_id
    wandb_logger.experiment.config["node_features"] = node_features

    if model_name is not None:
        wandb_logger.experiment.config["model_name"] = model_name
