import os
import sys

sys.path.append(os.curdir)

from typing import Dict
import pandas as pd
from metrics.tasks import TaskType
from models.models import ModelType
from datasets.transforms import TransformType
from datasets.dataset_types import DatasetType
from experiments.utils.results_processing.utils import to_markdown_file
from experiments.utils.results_processing.per_task import per_task
from experiments.utils.results_processing.per_model import per_model
from experiments.utils.results_processing.per_transform import per_transform
from experiments.utils.results_processing.per_family import per_family
from experiments.utils.results_processing.per_barycentric_subdv import (
    per_barycentric_subdivision,
)
from experiments.utils.results_processing.per_transform_and_family import (
    per_transform_and_family,
)
from experiments.utils.results_processing.utils import (
    read_result_csv,
    filter_for_ds_type,
)
import argparse


def process_results(
    results: Dict[TaskType, pd.DataFrame],
    include_topo_models: bool = True,
    include_name_task: bool = True,
    max_info: bool = False,
):
    tasks = (
        [TaskType.BETTI_NUMBERS, TaskType.NAME, TaskType.ORIENTABILITY]
        if include_name_task
        else [TaskType.BETTI_NUMBERS, TaskType.ORIENTABILITY]
    )
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

    models_2 = (
        [ModelType.SAN, ModelType.SCCN, ModelType.SCN, ModelType.SCCNN]
        if include_topo_models
        else []
    )
    transforms_2 = (
        [
            TransformType.degree_transform_sc,
            TransformType.random_simplices_features,
        ]
        if include_topo_models
        else []
    )

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
        max_info=max_info,
    )

    per_transform_res = per_transform(
        tasks=tasks,
        model_types_cartesian=[models, models_2],
        transform_types_cartesian=[transforms, transforms_2],
        result_dataframes=results,
        max_info=max_info,
    )

    return per_task_res, per_model_res, per_transform_res


def process_ds_type(
    ds_type: DatasetType,
    include_topo_models: bool = True,
    include_name_task: bool = True,
    max_info: bool = False,
    csv_prefix: str = "results_",
):
    results_csv_prefix = csv_prefix
    results = read_result_csv(results_csv_prefix)

    results_filtered = {
        task_type: filter_for_ds_type(res_task, ds_type)
        for task_type, res_task in results.items()
    }

    per_task, per_model, per_transform = process_results(
        results_filtered,
        include_topo_models,
        include_name_task=include_name_task,
        max_info=max_info,
    )

    mkd_str = f"# Overview - {ds_type.name}\n"
    mkd_str += to_markdown_file(per_task)

    mkd_str += "# Per Model\n"
    mkd_str += to_markdown_file(per_model)

    mkd_str += "# Per Transform\n"
    mkd_str += to_markdown_file(per_transform)

    with open(f"results_{ds_type.name.lower()}.md", "w") as file:
        file.write(mkd_str)


def process(max_info: bool = False, csv_prefix: str = "results_"):
    process_ds_type(
        DatasetType.FULL_2D,
        include_topo_models=True,
        include_name_task=True,
        max_info=max_info,
        csv_prefix=csv_prefix,
    )
    process_ds_type(
        DatasetType.NO_NAMELESS_2D,
        include_topo_models=True,
        include_name_task=True,
        max_info=max_info,
        csv_prefix=csv_prefix,
    )
    process_ds_type(
        DatasetType.FULL_3D,
        include_topo_models=False,
        include_name_task=False,
        max_info=max_info,
        csv_prefix=csv_prefix,
    )


def process_barycentric(csv_prefix: str):
    results = read_result_csv(csv_prefix)
    task_results = per_barycentric_subdivision(result_dataframes=results)
    for task_result in task_results:
        task_name, task_result_data = task_result
        task_result_data: pd.DataFrame = task_result_data
        r_path = f"{task_name}_barycentric_subdivisions.csv"
        task_result_data.to_csv(r_path)


def process_per_family(csv_prefix: str):
    results = read_result_csv(csv_prefix)
    for ds_type in DatasetType:
        results_filtered = {
            task_type: filter_for_ds_type(res_task, ds_type)
            for task_type, res_task in results.items()
        }
        task_results = per_family(results_filtered, ds_type)
        for task_result in task_results:
            task_name, task_result_data = task_result
            task_result_data: pd.DataFrame = task_result_data
            r_path = f"{task_name}_{ds_type.name.lower()}_per_family.csv"
            task_result_data.to_csv(r_path)


def process_per_transform_and_family(csv_prefix: str):
    results = read_result_csv(csv_prefix)
    results_filtered = {
        task_type: filter_for_ds_type(res_task, DatasetType.FULL_2D)
        for task_type, res_task in results.items()
    }
    task_results = per_transform_and_family(result_dataframes=results_filtered)
    for task_result in task_results:
        task_name, task_result_data = task_result
        task_result_data: pd.DataFrame = task_result_data
        r_path = f"{task_name}_per_transform_and_family.csv"
        task_result_data.to_csv(r_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Argument parser for configurations."
    )
    parser.add_argument(
        "--max_info",
        action="store_true",
        help="Whether to add which model/ transform achieved max performance.",
    )
    parser.add_argument(
        "--csv_prefix",
        type=str,
        help="Maximum number of barycentric subdivisions to perform for test evaluation.",
        default="results_",
    )
    parser.add_argument(
        "--barycentric",
        action="store_true",
        help="Whether the results originate from the barycentric subdivisions experiment.",
    )
    parser.add_argument(
        "--for_submission",
        action="store_true",
        help="Table generation for final paper submission.",
    )
    args = parser.parse_args()

    max_info: bool = args.max_info
    barycentric: bool = args.barycentric
    csv_prefix: str = args.csv_prefix
    for_submission: bool = args.for_submission

    if barycentric:
        process_barycentric(csv_prefix=csv_prefix)
    elif for_submission:
        process_per_transform_and_family(csv_prefix=csv_prefix)
        process_per_family(csv_prefix=csv_prefix)
    else:
        process(max_info=max_info, csv_prefix=csv_prefix)
