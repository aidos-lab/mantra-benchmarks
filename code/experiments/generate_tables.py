import sys
import os

sys.path.append(os.curdir)

from typing import List, Dict
import pandas as pd
from metrics.tasks import TaskType
from models.models import ModelType
from datasets.transforms import TransformType
from experiments.utils.results_processing.utils import to_markdown_file
from experiments.utils.results_processing.per_task import per_task
from experiments.utils.results_processing.per_model import per_model
from experiments.utils.results_processing.per_transform import per_transform
from experiments.utils.results_processing.utils import read_result_csv

def process_name_orientability(results: Dict[TaskType, pd.DataFrame]):
    tasks = [TaskType.NAME, TaskType.ORIENTABILITY]
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

    # --------------------------------------------------------------------------

    per_task_res = per_task(
        tasks=tasks,
        model_types_cartesian=[models],
        transform_types_cartesian=[transforms],
        result_dataframes=results
    )

    per_model_res = per_model(
        tasks=tasks,
        model_types_cartesian=[models],
        transform_types_cartesian=[transforms],
        result_dataframes=results
    )

    per_transform_res = per_transform(
        tasks=tasks,
        model_types_cartesian=[models],
        transform_types_cartesian=[transforms],
        result_dataframes=results
    )
    return per_task_res, per_model_res, per_transform_res


def process_betti(results: Dict[TaskType, pd.DataFrame]):
    tasks = [TaskType.BETTI_NUMBERS]
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
        result_dataframes=results
    )

    per_model_res = per_model(
        tasks=tasks,
        model_types_cartesian=[models, models_2],
        transform_types_cartesian=[transforms, transforms_2],
        result_dataframes=results
    )

    per_transform_res = per_transform(
        tasks=tasks,
        model_types_cartesian=[models, models_2],
        transform_types_cartesian=[transforms, transforms_2],
        result_dataframes=results
    )

    return per_task_res, per_model_res, per_transform_res


def process():
    results_csv_prefix = "results_"
    results =read_result_csv(results_csv_prefix)

    res_betti = process_betti(results)
    res_name_or = process_name_orientability(results)

    per_task = res_betti[0] + res_name_or[0]
    per_model = res_betti[1] + res_name_or[1]
    per_transform = res_betti[2] + res_name_or[2]

    mkd_str = "# Overview\n"
    mkd_str += to_markdown_file(per_task)

    mkd_str += "# Per Model\n"
    mkd_str += to_markdown_file(per_model)

    mkd_str += "# Per Transform\n"
    mkd_str += to_markdown_file(per_transform)

    with open("results.md", "w") as file:
        file.write(mkd_str)


if __name__ == "__main__":
    process()
