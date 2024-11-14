from typing import Dict

import numpy as np
import pandas as pd

from datasets.dataset_types import DatasetType
from datasets.transforms import (
    TransformType,
)
from metrics.tasks import TaskType
from models.model_types import simplicial_models, graphbased_models
from .utils import get_matching_indeces, get_metric_col_names, format_res_val


def per_family(
        result_dataframes: Dict[TaskType, pd.DataFrame], ds_type: DatasetType
):
    model_families = [
        {
            "models": simplicial_models,
            "name": "simplicial",
        },
        {
            "models": graphbased_models,
            "name": "graphbased",
        },
    ]
    results = []

    for task_type in TaskType:
        if task_type == TaskType.NAME and ds_type == DatasetType.FULL_3D:
            continue
        df = result_dataframes[task_type]

        metric_cols = get_metric_col_names(df, task_type)
        res_cols = ["Family"] + [metric[5:] for metric in metric_cols]
        df_results = pd.DataFrame(columns=res_cols)

        for model_family in model_families:

            row = {
                "Model": model_family["name"],
            }

            for metric in metric_cols:
                benchmarks = []
                for tr in TransformType:

                    for model_type in model_family["models"]:
                        filtered_results = df[
                            get_matching_indeces(df, model_type, tr)
                        ]
                        filtered_results_ = filtered_results[metric].to_numpy()
                        if len(filtered_results_) > 1:
                            benchmarks.append(filtered_results_.max())
                if len(benchmarks) > 1:
                    benchmarks = np.asarray(benchmarks)
                    row[metric[5:]] = format_res_val(
                        np.mean(benchmarks),
                        np.std(benchmarks),
                        note=f"max:{np.max(benchmarks):.2f}",
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
