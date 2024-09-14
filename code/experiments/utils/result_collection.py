from typing import List, Dict
from metrics.tasks import TaskType
from experiments.utils.configs import ConfigExperimentRun
import pandas as pd


class Result:
    def __init__(
        self, data: Dict[str, float], config: ConfigExperimentRun
    ) -> None:
        self.data = data
        self.config = config
        self.ds_type = config.ds_type


class ResultCollection:
    collection: Dict[TaskType, List[Result]]

    def __init__(self) -> None:
        self.collection = {
            TaskType.BETTI_NUMBERS: [],
            TaskType.NAME: [],
            TaskType.ORIENTABILITY: [],
        }

    def add(self, data: Dict[str, float], config: ConfigExperimentRun):
        self.collection[config.task_type].append(
            Result(data=data, config=config)
        )

    def save(self, t_file_prefix: str = "res"):

        for task_type in TaskType:
            t_file = f"{t_file_prefix}_{task_type.name.lower()}.csv"

            data = []
            for x in self.collection[task_type]:
                result = x.data
                result["type_model"] = x.config.conf_model.type.name.lower()
                result["transform"] = x.config.transforms.name.lower()
                result["ds_type"] = x.config.ds_type.name.lower()
                data.append(result)

            df = pd.DataFrame(data)
            df.to_csv(t_file, index=False)
