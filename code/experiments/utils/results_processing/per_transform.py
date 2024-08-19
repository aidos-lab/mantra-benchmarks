from typing import List, Tuple
import pandas as pd
from metrics.tasks import TaskType
from models.models import ModelType
from datasets.transforms import TransformType
import numpy as np
from .utils import (
    get_matching_indeces,
    get_metric_col_names,
    get_result_path,
    format_res_val,
)


def reduce(
    df: pd.DataFrame,
    model_types: List[ModelType],
    transform_types: List[TransformType],
    row: any,
    metric: str,
):
    """
    Reduce the dataframe by aggregating metric results for each transform type and model type.

    Parameters:
    df (DataFrame): The input dataframe containing the results.
    model_types (List[ModelType]): List of model types to process.
    transform_types (List[TransformType]): List of transform types to process.
    row (dict): Dictionary to append the results.
    metric (str): The metric column to process.

    Returns:
    dict: The dictionary with the aggregated results.
    """

    for transform_type in transform_types:
        results = []

        for model_type in model_types:
            filtered_results = df[
                get_matching_indeces(df, model_type, transform_type)
            ]
            results.append(filtered_results[metric].max())

        row[transform_type.name.lower()] = format_res_val(
            np.max(results), np.std(results)
        )

    return row


def per_transform(
    tasks: List[TaskType],
    model_types_cartesian: List[List[ModelType]],
    transform_types_cartesian: List[List[TransformType]],
    result_csv_prefix: str = "./ignore_temp_",
) -> List[Tuple[str, pd.DataFrame]]:
    """
    Generate results per transform type for each task type by processing model and transform type combinations.

    Parameters:
    tasks (List[TaskType]): List of task types to process.
    model_types_cartesian (List[List[ModelType]]): Model types to process.
    transform_types_cartesian (List[List[TransformType]]): Transform types to process.
    result_csv_prefix (str): Prefix for the result CSV file paths.

    Returns:
    List[Tuple[str, DataFrame]]: List of tuples containing task type names and their corresponding result dataframes.
    """

    results = []
    all_transforms = []
    for transforms in transform_types_cartesian:
        all_transforms = all_transforms + transforms

    n_cartesian = len(model_types_cartesian)
    results = []
    for task_type in tasks:
        result_path = get_result_path(result_csv_prefix, task_type)
        df = pd.read_csv(result_path)
        metric_cols = get_metric_col_names(df, task_type)

        res_cols = ["Metric"] + [
            transform.name.lower() for transform in all_transforms
        ]

        df_results = pd.DataFrame(columns=res_cols)

        for metric in metric_cols:
            row = {"Metric": metric}
            for i_cartesian in range(n_cartesian):
                row = reduce(
                    df,
                    model_types=model_types_cartesian[i_cartesian],
                    transform_types=transform_types_cartesian[i_cartesian],
                    row=row,
                    metric=metric,
                )
            new_row_df = pd.DataFrame([row])
            concat_df = (
                [df_results, new_row_df]
                if len(df_results) > 0
                else [new_row_df]
            )
            df_results = pd.concat(concat_df)

        results.append((task_type.name.lower(), df_results))

    return results
