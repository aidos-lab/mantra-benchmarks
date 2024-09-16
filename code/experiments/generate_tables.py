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
):
    results_csv_prefix = "results_"
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


def process(max_info: bool = False):
    process_ds_type(
        DatasetType.FULL_2D,
        include_topo_models=True,
        include_name_task=True,
        max_info=max_info,
    )
    process_ds_type(
        DatasetType.NO_NAMELESS_2D,
        include_topo_models=True,
        include_name_task=True,
        max_info=max_info,
    )
    process_ds_type(
        DatasetType.FULL_3D,
        include_topo_models=False,
        include_name_task=False,
        max_info=max_info,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Argument parser for configurations."
    )
    parser.add_argument(
        "--max_info",
        action="store_true",
        help="Whether to add which model/ transform achieved max performance.",
    )
    args = parser.parse_args()

    max_info: bool = args.max_info
    process(max_info=max_info)
