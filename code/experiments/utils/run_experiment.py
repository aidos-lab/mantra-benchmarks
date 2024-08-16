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
from typing import List
from .imbalance_handling import sorted_imbalance_weights
import os


def get_setup(
    config: ConfigExperimentRun,
    use_logger: bool = True,
    data_dir: str = "./data",
) -> Tuple[SimplicialDataModule, BaseModel, L.Trainer, WandbLogger]:
    run_id = str(uuid.uuid4())
    transforms = transforms_lookup[config.transforms]
    task_lookup: Dict[TaskType, Task] = get_task_lookup(transforms)

    dm = SimplicialDataModule(
        data_dir=data_dir,
        transform=task_lookup[config.task_type].transforms,
        use_stratified=config.use_stratified,
        task_type=config.task_type,
        seed=config.seed,
        dataloader_builder=dataloader_lookup[config.conf_model.type],
    )

    # ignore imbalance when working with betti numbers
    imbalance = [1]
    if config.task_type != TaskType.BETTI_NUMBERS:
        imbalance = sorted_imbalance_weights(
            dm.class_imbalance_statistics(), config.task_type
        )
        print("[INFO] Using imbalance weights for weighted loss: ", imbalance)

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
    if use_logger:
        print(data_dir)
        print("AASDDFASDFAS")
        logger = get_wandb_logger(
            save_dir=os.path.join(data_dir, "lightning_logs"),
            task_name=config.task_type.name,
            model_name=config.conf_model.type.name,
            node_features=config.transforms.name,
            run_id=run_id,
            project_id=config.logging.wandb_project_id,
        )
    else:
        logger = True

    trainer = L.Trainer(
        logger=logger,
        accelerator=config.trainer_config.accelerator,
        max_epochs=config.trainer_config.max_epochs,
        log_every_n_steps=config.trainer_config.log_every_n_steps,
        fast_dev_run=False,
        default_root_dir=data_dir,
    )

    return dm, lit_model, trainer, logger


def run_configuration(
    config: ConfigExperimentRun,
    save_checkpoint_path: Optional[str] = None,
    data_dir: str = "./data",
):
    dm, lit_model, trainer, logger = get_setup(config, data_dir=data_dir)

    # run
    trainer.fit(lit_model, dm)
    logger.experiment.finish()

    if save_checkpoint_path:
        print(f"[INFO] Saving checkpoint here {save_checkpoint_path}")
        trainer.save_checkpoint(save_checkpoint_path)

    return trainer


def benchmark_configuration(
    config: ConfigExperimentRun,
    save_checkpoint_path: str,
    use_logger: bool = False,
    data_dir: str = "./data",
) -> List[Dict[str, float]]:
    dm, lit_model, trainer, logger = get_setup(
        config, use_logger=use_logger, data_dir=data_dir
    )

    output = trainer.test(lit_model, dm, save_checkpoint_path)

    if use_logger:
        logger.experiment.finish()
    return output
