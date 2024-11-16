"""
Code for running the cell mp
"""

from metrics.tasks import (
    TaskType,
    Task,
    get_task_lookup,
)
from experiments.utils.configs import ConfigExperimentRun, TrainerConfig
from models import model_lookup
from metrics.tasks import TaskType
from typing import Dict
from datasets.simplicial import SimplicialDataModule
from models.base import BaseModel
from experiments.utils.loggers import get_wandb_logger, update_wandb_logger
import lightning as L
import uuid
from datasets.transforms import transforms_lookup
from models.models import dataloader_lookup
from typing import Dict
from experiments.utils.imbalance_handling import sorted_imbalance_weights
import os
from datasets.dataset_types import DatasetType
from datasets.transforms import TransformType
from models.cells.mp.cin0 import CellMPConfig


# CONFIG ---------------------------------------------------------------------

# model
model_config = CellMPConfig(
    num_input_features=1,
    num_classes=3,
)
transform_type = TransformType.degree_transform_sc
# model_config = MLPConfig(
#     num_hidden_neurons=64,
#     num_hidden_layers=3,
#     num_node_features=10,
#     out_channels=3
# )
# transform_type = TransformType.degree_transform_onehot

# trainer
trainer_config = TrainerConfig(max_epochs=10, log_every_n_steps=1)

# full config
config = ConfigExperimentRun(
    seed=10,
    ds_type=DatasetType.FULL_2D,
    transforms=TransformType.degree_transform_sc,
    use_stratified=True,
    task_type=TaskType.BETTI_NUMBERS,
    trainer_config=trainer_config,
    conf_model=model_config,
)

# data and logging
data_dir = "/data"
use_logger = False
devices = [0]
run_id = str(uuid.uuid4())

# ----------------------------------------------------------------------------

# SETUP ----------------------------------------------------------------------

# logging
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

# transforms
transforms = transforms_lookup(config.transforms, config.ds_type)
task_lookup: Dict[TaskType, Task] = get_task_lookup(
    transforms, ds_type=config.ds_type
)
dataset_transforms = task_lookup[config.task_type].transforms

# data module
dm = SimplicialDataModule(
    ds_type=config.ds_type,
    data_dir=data_dir,
    transform=dataset_transforms,
    use_stratified=config.use_stratified,
    task_type=config.task_type,
    seed=config.seed,
    dataloader_builder=dataloader_lookup[config.conf_model.type],
    num_barycentric_subdivisions=0,
)

# imbalance weighting
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

# instantiate model
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


trainer = L.Trainer(
    logger=logger,
    accelerator=config.trainer_config.accelerator,
    max_epochs=config.trainer_config.max_epochs,
    log_every_n_steps=config.trainer_config.log_every_n_steps,
    fast_dev_run=False,
    default_root_dir=data_dir,
    devices=devices,
    # strategy='ddp_find_unused_parameters_true'
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
# ----------------------------------------------------------------------------

# RUN ------------------------------------------------------------------------
trainer.fit(lit_model, dm)
# ----------------------------------------------------------------------------
