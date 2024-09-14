from typing import List, Dict
from metrics.tasks import TaskType
from experiments.utils.configs import ConfigExperimentRun
import pandas as pd


class Result:
    def __init__(
        self,
        data: Dict[str, float],
        config: ConfigExperimentRun,
        barycentric_subdivision_idx: int = 0,
    ) -> None:
        self.data = data
        self.config = config
        self.ds_type = config.ds_type
        self.barycentric_subdivision_idx = barycentric_subdivision_idx


class ResultCollection:
    collection: Dict[TaskType, List[Result]]

    def __init__(self) -> None:
        self.collection = {
            TaskType.BETTI_NUMBERS: [],
            TaskType.NAME: [],
            TaskType.ORIENTABILITY: [],
        }

    def add(
        self,
        data: Dict[str, float],
        config: ConfigExperimentRun,
        barycentric_subdivision_idx: int = 0,
    ):
        self.collection[config.task_type].append(
            Result(
                data=data,
                config=config,
                barycentric_subdivision_idx=barycentric_subdivision_idx,
            )
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
                result["barycentric_subdivision_idx"] = (
                    x.barycentric_subdivision_idx
                )
                data.append(result)

            df = pd.DataFrame(data)
            df.to_csv(t_file, index=False)
