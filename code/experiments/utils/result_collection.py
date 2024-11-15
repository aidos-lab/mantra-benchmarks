import os
from typing import List, Dict

import pandas as pd

from datasets.dataset_types import DatasetType
from datasets.transforms import TransformType
from experiments.utils.configs import ConfigExperimentRun
from experiments.utils.enum_utils import enum_from_str_id
from metrics.tasks import TaskType
from models import model_cfg_lookup
from models.model_types import ModelType


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

    def load(self, t_file_prefix: str = "res"):

        for task_type in TaskType:
            t_file = f"{t_file_prefix}_{task_type.name.lower()}.csv"

            if not os.path.exists(t_file):
                continue

            try:
                df = pd.read_csv(t_file)
            except pd.errors.EmptyDataError:
                continue

            print(f"[INFO] Loading existing {t_file}")

            entries_per_task = []

            for _, row in df.iterrows():
                # read csv and reconstruct entry
                r_data = row.drop(
                    [
                        "type_model",
                        "transform",
                        "ds_type",
                        "barycentric_subdivision_idx",
                    ]
                ).to_dict()
                r_bary_subd_idx = row["barycentric_subdivision_idx"]
                model_type = enum_from_str_id(row["type_model"], ModelType)
                conf_model = model_cfg_lookup[model_type]()
                r_config = ConfigExperimentRun(conf_model=conf_model)
                r_config.transforms = enum_from_str_id(
                    row["transform"], TransformType
                )
                r_config.ds_type = enum_from_str_id(
                    row["ds_type"], DatasetType
                )
                r_config.task_type = task_type

                x = Result(
                    data=r_data,
                    config=r_config,
                    barycentric_subdivision_idx=r_bary_subd_idx,
                )

                entries_per_task.append(x)

            self.collection[task_type] = entries_per_task

    def exists(
        self, config: ConfigExperimentRun, barycentric_subdivision_idx
    ) -> int:
        """
        Returns the number of matching entries in the result collection.
        """

        return sum(
            res.config.conf_model.type == config.conf_model.type
            and res.config.transforms == config.transforms
            and res.config.ds_type == config.ds_type
            and res.barycentric_subdivision_idx == barycentric_subdivision_idx
            for res in self.collection[config.task_type]
        )
