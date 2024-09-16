from typing import List, Tuple, Dict
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
    df_results: pd.DataFrame,
    metric_columns,
):
    """
    Reduce the dataframe by aggregating metric results for each model type and transform type.

    Parameters:
    df (DataFrame): The input dataframe containing the results.
    model_types (List[ModelType]): List of model types to process.
    transform_types (List[TransformType]): List of transform types to process.
    df_results (DataFrame): The dataframe to append the results.
    metric_columns (List[str]): List of metric columns to process.

    Returns:
    DataFrame: The dataframe with the aggregated results.
    """
    for model_type in model_types:

        row = {"Model": model_type.name.lower()}

        for metric in metric_columns:

            metric_results = []

            for transform_type in transform_types:
                filtered_results = df[
                    get_matching_indeces(df, model_type, transform_type)
                ]
                metric_results.append(filtered_results[metric].max())
            row[metric[5:]] = format_res_val(
                np.max(metric_results), np.std(metric_results)
            )

        new_row_df = pd.DataFrame([row])
        concat_df = (
            [df_results, new_row_df] if len(df_results) > 0 else [new_row_df]
        )
        df_results = pd.concat(concat_df)
    return df_results


def per_model(
    tasks: List[TaskType],
    model_types_cartesian: List[List[ModelType]],
    transform_types_cartesian: List[List[TransformType]],
    result_dataframes: Dict[TaskType, pd.DataFrame],
) -> List[Tuple[str, pd.DataFrame]]:
    """
    Generate results per model for each task type by processing model and transform type combinations.

    Parameters:
    tasks (List[TaskType]): List of task types to process.
    model_types_cartesian (List[List[ModelType]]): Model types to process.
    transform_types_cartesian (List[List[TransformType]]): Transform types to process.
    result_dataframes (Dict[TaskType, pd.DataFrame]): Dataframes per task.

    Returns:
    List[Tuple[str, DataFrame]]: List of tuples containing task type names and their corresponding result dataframes.
    """
    n_cartesian = len(model_types_cartesian)
    results = []
    for task_type in tasks:
        df = result_dataframes[task_type]
        metric_cols = get_metric_col_names(df, task_type)

        res_cols = ["Model"] + [metric[5:] for metric in metric_cols]
        df_results = pd.DataFrame(columns=res_cols)

        for i_cartesian in range(n_cartesian):
            df_results = reduce(
                df,
                transform_types=transform_types_cartesian[i_cartesian],
                model_types=model_types_cartesian[i_cartesian],
                df_results=df_results,
                metric_columns=metric_cols,
            )
        results.append((task_type.name.lower(), df_results))
    return results
