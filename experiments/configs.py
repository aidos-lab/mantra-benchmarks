"""
Pydantic configuration of experiment runs.
"""

from pydantic import Discriminator, Field, ValidationError
from pydantic_settings import BaseSettings
from models.models import ModelConfig
from datasets.transforms import TransformType
from metrics.tasks import TaskType
from models.models import ModelType
import yaml
from typing import Any, List


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
    transforms: TransformType = TransformType.degree_transform_onehot
    use_stratified: bool = True
    logging: WandbConfig = WandbConfig()
    task_type: TaskType = TaskType.BETTI_NUMBERS
    learning_rate: float = 1e-3
    trainer_config: TrainerConfig = TrainerConfig()
    conf_model: ModelConfig = Field(
        discriminator=Discriminator(get_discriminator_value)
    )


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
