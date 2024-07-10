from typing import List, Tuple
import pandas as pd
from metrics.tasks import TaskType
from models.models import ModelType
from datasets.transforms import TransformType


def get_matching_indeces(
    df: pd.DataFrame, model_type: ModelType, transform_type: TransformType
):
    """
    Get indices of rows in the dataframe that match the specified model type and transform type.

    Parameters:
    df (DataFrame): The input dataframe containing the results.
    model_type (ModelType): The model type to match.
    transform_type (TransformType): The transform type to match.

    Returns:
    Series: Boolean series indicating rows that match the specified criteria.
    """
    return (df["type_model"] == model_type.name.lower()) & (
        df["transform"] == transform_type.name.lower()
    )


def get_metric_col_names(df: pd.DataFrame, task_type: TaskType):
    """
    Get column names in the dataframe that start with "test_" and correspond to the specified task type.

    Parameters:
    df (DataFrame): The input dataframe containing the results.
    task_type (TaskType): The task type to match.

    Returns:
    List[str]: List of column names that start with "test_".
    """
    return [
        col
        for col in df[(df["type_model"] == task_type.name.lower())].columns
        if col.startswith("test_") and col != "test_loss"
    ]


def get_result_path(result_csv_prefix: str, task_type: TaskType):
    """
    Generate the file path for the result CSV file based on the given prefix and task type.

    Parameters:
    result_csv_prefix (str): The prefix for the result CSV file.
    task_type (TaskType): The task type to include in the file name.

    Returns:
    str: The generated file path.
    """
    return f"./{result_csv_prefix}{task_type.name.lower()}.csv"


def to_markdown_file(results: List[Tuple[str, pd.DataFrame]]):
    """
    Convert a list of dataframes to a markdown formatted string.

    Parameters:
    results (List[Tuple[str, DataFrame]]): List of tuples, each containing an identifier and a dataframe.

    Returns:
    str: A string with the markdown formatted dataframes.
    """
    mkd_str = ""
    for result in results:
        id, df = result
        mkd_str += f"## {id}\n"
        mkd_str += df.to_markdown()
        mkd_str += "\n"

    return mkd_str
