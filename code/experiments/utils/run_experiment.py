"""
Code for a single experiment run.
"""

import os
import uuid
from collections import ChainMap
from typing import List, Dict
from typing import Optional, Tuple

import lightning as L
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.loggers import WandbLogger

from datasets.simplicial import SimplicialDataModule
from datasets.transforms import (
    transforms_lookup,
    SimplicialComplexHodgeLapEigPETransform,
)
from experiments.utils.configs import ConfigExperimentRun
from experiments.utils.loggers import get_wandb_logger, update_wandb_logger
from metrics.tasks import (
    Task,
    get_task_lookup,
)
from metrics.tasks import TaskType
from models import model_lookup, ModelType
from models.base import BaseModel
from models.models import dataloader_lookup
from .imbalance_handling import sorted_imbalance_weights


def get_data_module(
    config: ConfigExperimentRun,
    data_dir: str = "./data",
    number_of_barycentric_subdivisions: int = 0,
) -> SimplicialDataModule:
    transforms = transforms_lookup(config.transforms, config.ds_type)
    # Hack: If we use the transformers, then we add the positional encodings
    if config.conf_model.type == ModelType.CELL_TRANSF:
        transforms.append(SimplicialComplexHodgeLapEigPETransform())
    task_lookup: Dict[TaskType, Task] = get_task_lookup(
        transforms, ds_type=config.ds_type
    )

    dataset_transforms = task_lookup[config.task_type].transforms

    dm = SimplicialDataModule(
        ds_type=config.ds_type,
        data_dir=data_dir,
        transform=dataset_transforms,
        use_stratified=config.use_stratified,
        task_type=config.task_type,
        seed=config.seed,
        dataloader_builder=dataloader_lookup[config.conf_model.type],
        num_barycentric_subdivisions=number_of_barycentric_subdivisions,
    )
    return dm


def get_setup(
    config: ConfigExperimentRun,
    use_logger: bool = True,
    data_dir: str = "./data",
    devices=None,
) -> Tuple[SimplicialDataModule, BaseModel, L.Trainer, WandbLogger]:
    run_id = str(uuid.uuid4())
    transforms = transforms_lookup(config.transforms, config.ds_type)
    task_lookup: Dict[TaskType, Task] = get_task_lookup(
        transforms, ds_type=config.ds_type
    )
    dm = get_data_module(config, data_dir)
    # ignore imbalance when working with betti numbers
    if (
        config.use_imbalance_weighting
        and config.task_type != TaskType.BETTI_NUMBERS
    ):
        imbalance = sorted_imbalance_weights(
            dm.class_imbalance_statistics(), config.task_type
        )
        print("[INFO] Using imbalance weights for weighted loss: ", imbalance)
    else:
        imbalance = None

    model = model_lookup[config.conf_model.type](config.conf_model)
    metrics = task_lookup[config.task_type].get_metrics(config.ds_type)
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
        devices=devices,
        strategy="ddp_find_unused_parameters_true",
    )

    if use_logger and trainer.global_rank == 0:
        update_wandb_logger(
            logger,
            task_name=config.task_type.name,
            model_name=config.conf_model.type.name,
            node_features=config.transforms.name,
            run_id=run_id,
            project_id=config.logging.wandb_project_id,
        )

    return dm, lit_model, trainer, logger


def run_configuration(
    config: ConfigExperimentRun,
    save_checkpoint_path: Optional[str] = None,
    data_dir: str = "./data",
    devices=None,
):
    dm, lit_model, trainer, logger = get_setup(
        config, data_dir=data_dir, devices=devices
    )

    # run
    trainer.fit(lit_model, dm)
    outp = trainer.test(lit_model, dm)
    logger.experiment.finish()

    if save_checkpoint_path:
        print(f"[INFO] Saving checkpoint here {save_checkpoint_path}")
        trainer.save_checkpoint(save_checkpoint_path)

    return outp


def retrieve_benchmarks(
    trainer: L.Trainer,
    model: LightningModule,
    dm: LightningDataModule,
    save_path: str,
    test_on_train_set: bool = True,
) -> Dict:
    out_test = trainer.test(model, dm, save_path)[0]
    if test_on_train_set:
        out_train = trainer.validate(model, dm.train_dataloader(), save_path)[
            0
        ]
        out_train = {
            key.replace("validation", "train"): item
            for key, item in out_train.items()
        }
        out = dict(ChainMap(out_test, out_train))
        return out
    else:
        return out_test


def benchmark_configuration(
    config: ConfigExperimentRun,
    save_checkpoint_path: str,
    use_logger: bool = False,
    data_dir: str = "./data",
    number_of_barycentric_subdivisions: int = 0,
    devices=None,
) -> List[Dict[str, float]]:
    dm, lit_model, trainer, logger = get_setup(
        config, use_logger=use_logger, data_dir=data_dir, devices=devices
    )

    benchmark_on_train_set = number_of_barycentric_subdivisions == 0
    outputs = [
        retrieve_benchmarks(
            trainer,
            lit_model,
            dm,
            save_checkpoint_path,
            test_on_train_set=benchmark_on_train_set,
        )
    ]
    for i in range(1, number_of_barycentric_subdivisions + 1):
        dm_bs = get_data_module(config, data_dir, i)
        outputs.append(
            retrieve_benchmarks(
                trainer,
                lit_model,
                dm_bs,
                save_checkpoint_path,
                test_on_train_set=benchmark_on_train_set,
            )
        )
    if use_logger:
        logger.experiment.finish()
    return outputs
