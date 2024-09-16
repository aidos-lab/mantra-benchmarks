import sys
import os

sys.path.append(os.curdir)

from typing import List, Dict
import pandas as pd
from metrics.tasks import TaskType
from models.models import ModelType
from datasets.transforms import TransformType
from datasets.dataset_types import DatasetType
from experiments.utils.results_processing.utils import to_markdown_file
from experiments.utils.results_processing.per_task import per_task
from experiments.utils.results_processing.per_model import per_model
from experiments.utils.results_processing.per_transform import per_transform
from experiments.utils.results_processing.utils import (
    read_result_csv,
    filter_for_ds_type,
)


def process_full2manifolds(results: Dict[TaskType, pd.DataFrame]):
    tasks = [TaskType.BETTI_NUMBERS, TaskType.NAME, TaskType.ORIENTABILITY]
    models = [
        ModelType.GAT,
        ModelType.GCN,
        ModelType.MLP,
        ModelType.TransfConv,
    ]
    transforms = [
        TransformType.degree_transform_onehot,
        TransformType.random_node_features,
        TransformType.degree_transform,
    ]

    models_2 = [ModelType.SAN, ModelType.SCCN, ModelType.SCN, ModelType.SCCNN]
    transforms_2 = [
        TransformType.degree_transform_sc,
        TransformType.random_simplices_features,
    ]

    per_task_res = per_task(
        tasks=tasks,
        model_types_cartesian=[models, models_2],
        transform_types_cartesian=[transforms, transforms_2],
        result_dataframes=results,
    )

    per_model_res = per_model(
        tasks=tasks,
        model_types_cartesian=[models, models_2],
        transform_types_cartesian=[transforms, transforms_2],
        result_dataframes=results,
    )

    per_transform_res = per_transform(
        tasks=tasks,
        model_types_cartesian=[models, models_2],
        transform_types_cartesian=[transforms, transforms_2],
        result_dataframes=results,
    )

    return per_task_res, per_model_res, per_transform_res


def process():
    results_csv_prefix = "results_"
    results = read_result_csv(results_csv_prefix)

    results_2d_full = {
        task_type: filter_for_ds_type(res_task, DatasetType.FULL_2D)
        for task_type, res_task in results.items()
    }

    per_task_full2, per_model_full2, per_transform_full2 = (
        process_full2manifolds(results_2d_full)
    )

    mkd_str = "# Overview\n"
    mkd_str += to_markdown_file(per_task_full2)

    mkd_str += "# Per Model\n"
    mkd_str += to_markdown_file(per_model_full2)

    mkd_str += "# Per Transform\n"
    mkd_str += to_markdown_file(per_transform_full2)

    with open("results_full2_manifolds.md", "w") as file:
        file.write(mkd_str)


if __name__ == "__main__":
    process()
