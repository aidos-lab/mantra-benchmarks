from typing import Literal

import lightning as L
import wandb
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader


def get_wandb_logger(
    task_name: Literal["orientability", "betti_numbers", "name"],
    save_dir="./lightning_logs",
    model_name: str = None,
):
    wandb_logger = WandbLogger(
        project="MANTRA", entity="aidos-labs", save_dir=save_dir
    )
    wandb_logger.experiment.config["task"] = task_name
    if model_name is not None:
        wandb_logger.experiment.config["model_name"] = model_name
    return wandb_logger


def perform_experiment(
    task: Literal["orientability", "betti_numbers", "name"],
    model,
    model_name,
    dataset,
    batch_size,
    num_workers,
    max_epochs,
    data_loader_class=DataLoader,
    accelerator="auto",
):
    if task == "orientability":
        train_indices = dataset.train_orientability_indices
        test_indices = dataset.test_orientability_indices
    elif task == "betti_numbers":
        train_indices = dataset.train_betti_numbers_indices
        test_indices = dataset.test_betti_numbers_indices
    elif task == "name":
        train_indices = dataset.train_name_indices
        test_indices = dataset.test_name_indices
    else:
        raise ValueError(f"Task {task} not recognized")
    logger = get_wandb_logger(task_name=task, model_name=model_name)
    train_ds = Subset(dataset, train_indices)
    test_ds = Subset(dataset, test_indices)
    train_dl = data_loader_class(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_dl = data_loader_class(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    trainer = L.Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=1,
        logger=logger,
        accelerator=accelerator,
    )
    trainer.fit(
        model,
        train_dl,
        test_dl,
    )
    wandb.finish()
