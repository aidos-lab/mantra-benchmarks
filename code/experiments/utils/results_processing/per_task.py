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
    metric_res_list: List[float],
    metric: str,
):
    """
    Reduce the results for a specific task by finding the maximum metric value for each model and transform type.

    Parameters:
    df (DataFrame): The input dataframe containing the results.
    model_types (List[ModelType]): List of model types to evaluate.
    transform_types (List[TransformType]): List of transform types to evaluate.
    metric_res_list (List[float]): List to store the resulting metric values.
    metric (str): The metric column name to evaluate.

    Returns:
    List[float]: Updated list of metric values with the maximum value for each model and transform type.
    """
    for model_type in model_types:

        model_results = []
        for transform_type in transform_types:
            filtered_results = df[
                get_matching_indeces(df, model_type, transform_type)
            ]
            max_ = filtered_results[metric].max()
            model_results.append(max_)
        metric_res_list.append(np.max(model_results))
    return metric_res_list


def per_task(
    tasks: List[TaskType],
    model_types_cartesian: List[List[ModelType]],
    transform_types_cartesian: List[List[TransformType]],
    result_dataframes: Dict[TaskType, pd.DataFrame],
) -> List[Tuple[str, pd.DataFrame]]:
    """
    Process results for each task by calculating the mean of the maximum metric values across different model and transform types.

    Parameters:
    tasks (List[TaskType]): List of tasks to evaluate.
    model_types_cartesian (List[List[ModelType]]): List of lists containing model types for each cartesian product.
    transform_types_cartesian (List[List[TransformType]]): List of lists containing transform types for each cartesian product.
    result_dataframes (Dict[TaskType, pd.DataFrame]): Dataframes per task.

    Returns:
    DataFrame: DataFrame containing processed results.
    """

    results = []
    n_cartesian = len(model_types_cartesian)
    assert n_cartesian == len(transform_types_cartesian)

    for task_type in tasks:
        df = result_dataframes[task_type]
        assert df is not None

        metric_col_names = get_metric_col_names(df, task_type)

        cols_result = ["Metric", "Mean"]
        df_results = pd.DataFrame(columns=cols_result)

        for metric in metric_col_names:

            res_for_metric = []
            for i_cartesian in range(n_cartesian):
                model_types = model_types_cartesian[i_cartesian]
                transform_types = transform_types_cartesian[i_cartesian]
                res_for_metric = reduce(
                    df, model_types, transform_types, res_for_metric, metric
                )

            if metric != "test_loss":
                row = {
                    "Metric": metric,
                    "Mean": format_res_val(np.mean(res_for_metric)),
                }
                new_row_df = pd.DataFrame([row])
                concat_df = (
                    [df_results, new_row_df]
                    if len(df_results) > 0
                    else [new_row_df]
                )
                df_results = pd.concat(concat_df)
        results.append((task_type.name.lower(), df_results))
    return results
