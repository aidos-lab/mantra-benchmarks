from models.model_types import simplicial_models, graphbased_models
from typing import Dict
from metrics.tasks import TaskType
import pandas as pd
from datasets.transforms import (
    TransformType,
    graphbased_transforms,
    simplicial_transforms,
)
from .utils import get_matching_indeces, get_metric_col_names, format_res_val
import numpy as np
from typing import Dict, List


def per_transform_and_family(result_dataframes: Dict[TaskType, pd.DataFrame]):

    model_families = [
        {
            "models": simplicial_models,
            "name": "simplicial",
            "transforms": simplicial_transforms,
        },
        {
            "models": graphbased_models,
            "name": "graphbased",
            "transforms": graphbased_transforms,
        },
    ]
    results = []

    for task_type in TaskType:
        df = result_dataframes[task_type]
        metric_cols = get_metric_col_names(df, task_type)
        res_cols = ["Family", "bary_idx"] + [
            metric[5:] for metric in metric_cols
        ]
        df_results = pd.DataFrame(columns=res_cols)

        for model_family in model_families:
            transform_families: Dict[str, List[TransformType]] = model_family[
                "transforms"
            ]
            for transform_family_key in transform_families:
                transforms = transform_families[transform_family_key]
                row = {
                    "Model": model_family["name"],
                    "Transform": transform_family_key,
                }

                for metric in metric_cols:
                    benchmarks = []
                    for tr in transforms:

                        for model_type in model_family["models"]:
                            filtered_results = df[
                                get_matching_indeces(df, model_type, tr)
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
