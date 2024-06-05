from pydantic_settings import BaseSettings
from models.models import ModelConfig
from mantra import TransformType
from metrics.tasks import TaskType
from models.models import ModelType


class TrainerConfig(BaseSettings):
    accelerator: str
    max_epochs: int
    log_every_n_steps: int


class ConfigExperimentRun(BaseSettings):
    seed: int = 10
    transforms: TransformType
    use_stratified: bool = True
    type_model: ModelType
    task_type: TaskType
    learning_rate: float
    trainer_config: TrainerConfig
    conf_model: ModelConfig
