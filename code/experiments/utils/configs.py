"""
Pydantic configuration of experiment runs.
"""

from pydantic import Discriminator, Field
from pydantic_settings import BaseSettings
from models.models import ModelConfig
from datasets.transforms import TransformType
from datasets.dataset_types import DatasetType
from metrics.tasks import TaskType
from models.models import ModelType
import yaml
from typing import Any, List, Optional
import os


class TrainerConfig(BaseSettings):
    accelerator: str = "auto"
    max_epochs: int = 10
    log_every_n_steps: int = 1


def get_discriminator_value(v: Any) -> ModelType:
    type_str = None
    if isinstance(v, dict):
        type_str = v.get("type")
    else:
        type_str = getattr(v, "type")
    return ModelType(type_str)


class WandbConfig(BaseSettings):
    wandb_project_id: str = "mantra-dev"


class ConfigExperimentRun(BaseSettings):
    seed: int = 10
    ds_type: DatasetType = DatasetType.FULL_2D
    transforms: TransformType = TransformType.degree_transform_onehot
    use_stratified: bool = True
    logging: WandbConfig = WandbConfig()
    task_type: TaskType = TaskType.BETTI_NUMBERS
    learning_rate: float = 1e-3
    use_imbalance_weighting: bool = False
    trainer_config: TrainerConfig = TrainerConfig()
    conf_model: ModelConfig = Field(
        discriminator=Discriminator(get_discriminator_value)
    )

    def get_identifier(self):
        identifier = f"{self.ds_type.name.lower()}_{self.transforms.name.lower()}_{self.task_type.name.lower()}_{self.conf_model.type.name.lower()}_seed_{self.seed}"
        return identifier

    def get_checkpoint_path(self, base_folder: str, run: Optional[int] = 0):
        identifier = self.get_identifier()
        fname = f"{identifier}_run_{run}.ckpt"
        return os.path.join(base_folder, fname)


def load_config(config_fpath: str) -> ConfigExperimentRun:
    with open(config_fpath, "r") as file:
        data = yaml.safe_load(file)
    config = ConfigExperimentRun.model_validate(data)
    return config


def load_configs(config_fpaths: List[str]) -> List[ConfigExperimentRun]:
    configs = []
    for config_fpath in config_fpaths:
        config = load_config(config_fpath)
        configs.append(config)

    return configs
