from typing import Dict

import numpy as np
import pandas as pd

from datasets.transforms import TransformType
from metrics.tasks import TaskType
from models.model_types import simplicial_models, graphbased_models
from .utils import get_matching_indeces, get_metric_col_names, format_res_val


def per_barycentric_subdivision(
    result_dataframes: Dict[TaskType, pd.DataFrame]
):
    model_families = [
        {"models": simplicial_models, "name": "simplicial"},
        {"models": graphbased_models, "name": "graphbased"},
    ]
    subdv_indices = [0, 1]
    results = []

    for task_type in TaskType:
        df = result_dataframes[task_type]
        metric_cols = get_metric_col_names(df, task_type)
        res_cols = ["Family", "bary_idx"] + [
            metric[5:] for metric in metric_cols
        ]
        df_results = pd.DataFrame(columns=res_cols)

        for subdv_idx in subdv_indices:

            for model_family in model_families:

                row = {"Model": model_family["name"], "bary_idx": subdv_idx}

                for metric in metric_cols:
                    benchmarks = []

                    for model_type in model_family["models"]:
                        for tr_type in TransformType:
                            filtered_results = df[
                                get_matching_indeces(
                                    df, model_type, tr_type, subdv_idx
                                )
                            ]
                            filtered_results_ = filtered_results[
                                metric
                            ].to_numpy()
                            if len(filtered_results_) > 1:
                                benchmarks.append(filtered_results_.max())

                    benchmarks = np.asarray(benchmarks)

                    row[metric[5:]] = format_res_val(
                        np.mean(benchmarks),
                        np.std(benchmarks),
                        note=f"max:{np.max(benchmarks) * 100:.2f}",
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
