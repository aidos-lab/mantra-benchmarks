"""
Code for a single experiment run.
"""

from metrics.tasks import (
    TaskType,
    Task,
    get_task_lookup,
)
from experiments.utils.configs import ConfigExperimentRun
from models import model_lookup
from metrics.tasks import TaskType
from typing import Dict, Optional, Tuple, List
from datasets.simplicial import SimplicialDataModule
from models.base import BaseModel
from experiments.utils.loggers import get_wandb_logger
import lightning as L
import uuid
from datasets.transforms import transforms_lookup
from lightning.pytorch.loggers import WandbLogger
from models.models import dataloader_lookup


def get_setup(
    config: ConfigExperimentRun,
) -> Tuple[SimplicialDataModule, BaseModel, L.Trainer, WandbLogger]:
    run_id = str(uuid.uuid4())
    transforms = transforms_lookup[config.transforms]
    task_lookup: Dict[TaskType, Task] = get_task_lookup(transforms)

    dm = SimplicialDataModule(
        data_dir="./data",
        transform=task_lookup[config.task_type].transforms,
        use_stratified=config.use_stratified,
        seed=config.seed,
        dataloader_builder=dataloader_lookup[config.conf_model.type],
    )

    imbalance_statistics = dm.class_imbalance_statistics()
    name_imbalance = imbalance_statistics["name"][1]
    orientability_imbalace = imbalance_statistics["orientable"][1]

    if config.task_type == TaskType.BETTI_NUMBERS:
        # betti numbers is linear regression, so no imbalance necessary
        imbalance = [1]
    elif config.task_type == TaskType.NAME:
        imbalance = name_imbalance
    elif config.task_type == TaskType.ORIENTABILITY:
        imbalance = orientability_imbalace

    print(imbalance)

    model = model_lookup[config.conf_model.type](config.conf_model)
    metrics = task_lookup[config.task_type].get_metrics()

    lit_model = BaseModel(
        model=model,
        training_accuracy=metrics.train,
        test_accuracy=metrics.test,
        validation_accuracy=metrics.val,
        accuracies_fn=task_lookup[config.task_type].accuracies,
        loss_fn=task_lookup[config.task_type].loss_fn,
        learning_rate=config.learning_rate,
        imbalance=imbalance,
    )

    logger = get_wandb_logger(
        task_name=config.task_type.name,
        model_name=config.conf_model.type.name,
        node_features=config.transforms.name,
        run_id=run_id,
        project_id=config.logging.wandb_project_id,
    )

    trainer = L.Trainer(
        logger=logger,
        accelerator=config.trainer_config.accelerator,
        max_epochs=config.trainer_config.max_epochs,
        log_every_n_steps=config.trainer_config.log_every_n_steps,
        fast_dev_run=False,
    )

    return dm, lit_model, trainer, logger


def run_configuration(
    config: ConfigExperimentRun, save_checkpoint_path: Optional[str] = None
):
    dm, lit_model, trainer, logger = get_setup(config)

    # run
    trainer.fit(lit_model, dm)
    logger.experiment.finish()

    if save_checkpoint_path:
        print(f"[INFO] Saving checkpoint here {save_checkpoint_path}")
        trainer.save_checkpoint(save_checkpoint_path)

    return trainer


def benchmark_configuration(
    config: ConfigExperimentRun, save_checkpoint_path: str
) -> List[Dict[str, float]]:
    dm, lit_model, trainer, logger = get_setup(config)

    output = trainer.test(lit_model, dm, save_checkpoint_path)
    logger.experiment.finish()
    return output
