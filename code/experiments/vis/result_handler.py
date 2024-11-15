from typing import Tuple, List

import pandas as pd
import wandb


def setup_wandb(wandb_project_id: str = "mantra-dev-run-3"):
    wandb.login()
    api = wandb.Api()
    runs = api.runs(wandb_project_id)

    full_history, config_list, name_list = [], [], []
    for run in runs:
        h = run._full_history()
        h = [r | run.config for r in h]
        full_history.extend(h)

    import json

    with open("raw_results.json", "w") as f:
        json.dump(full_history, f)
    return full_history


def convert_history(full_history):
    df = pd.DataFrame(full_history)
    df = df.set_index(
        ["task", "model_name", "node_features", "run_id"], inplace=False
    )
    mean_df = df.groupby(level=[0, 1, 2]).mean()
    mean_df = mean_df.add_suffix("_mean")

    std_df = df.groupby(level=[0, 1, 2]).std()
    std_df = std_df.add_suffix("_std")

    res_df = pd.concat([mean_df, std_df], join="outer", axis=1)

    res_df.to_csv("results.csv")

    df__ = res_df[
        [
            "validation_accuracy_mean",
            "validation_accuracy_std",
            "train_accuracy_mean",
            "train_accuracy_std",
            "validation_accuracy_betti_0_mean",
            "validation_accuracy_betti_0_std",
            "validation_accuracy_betti_1_mean",
            "validation_accuracy_betti_1_std",
            "validation_accuracy_betti_2_mean",
            "validation_accuracy_betti_2_std",
            "train_accuracy_betti_0_mean",
            "train_accuracy_betti_0_std",
            "train_accuracy_betti_1_mean",
            "train_accuracy_betti_1_std",
            "train_accuracy_betti_2_mean",
            "train_accuracy_betti_2_std",
        ]
    ]

    return df__


def process_df(df):
    reshaped_df = pd.DataFrame(
        columns=[
            "Task",
            "Model Name",
            "Node Features",
            "Mean Accuracy",
            "Std Accuracy",
            "Mean Train Accuracy",
            "Std Train Accuracy",
        ]
    )

    if "task" not in df.columns:
        df.reset_index(inplace=True)

    for index, row in df.iterrows():
        if pd.notna(row["validation_accuracy_betti_0_mean"]):
            val_acc_std_betti = (
                lambda i: f"validation_accuracy_betti_{int(i)}_std"
            )
            val_acc_mean_betti = (
                lambda i: f"validation_accuracy_betti_{int(i)}_mean"
            )
            train_acc_betti_mean = (
                lambda i: f"train_accuracy_betti_{int(i)}_std"
            )
            train_acc_betti_std = (
                lambda i: f"validation_accuracy_betti_{int(i)}_mean"
            )
            betti_task = lambda i: f"betti_{int(i)}"

            for i in range(3):
                new_row_dict = {
                    "Task": [betti_task(i)],
                    "Model Name": [row["model_name"]],
                    "Node Features": [row["node_features"]],
                    "Mean Accuracy": [row[val_acc_mean_betti(i)]],
                    "Std Accuracy": [row[val_acc_std_betti(i)]],
                    "Mean Train Accuracy": [row[train_acc_betti_mean(i)]],
                    "Std Train Accuracy": [row[train_acc_betti_std(i)]],
                }
                new_row = pd.DataFrame(new_row_dict, index=[0])
                reshaped_df = pd.concat(
                    [reshaped_df, new_row], ignore_index=True
                )
        else:
            new_row_dict = {
                "Task": [row["task"]],
                "Model Name": [row["model_name"]],
                "Node Features": [row["node_features"]],
                "Mean Accuracy": [row["validation_accuracy_mean"]],
                "Std Accuracy": [row["validation_accuracy_std"]],
                "Mean Train Accuracy": [row["train_accuracy_mean"]],
                "Std Train Accuracy": [row["train_accuracy_std"]],
            }
            new_row = pd.DataFrame(new_row_dict, index=[0])
            reshaped_df = pd.concat([reshaped_df, new_row], ignore_index=True)
    return reshaped_df


def pull_from_wandb():
    full_history = setup_wandb()
    df = convert_history(full_history)
    df = process_df(df)
    return df


class ResultHandler:
    df: pd.DataFrame

    def __init__(self, wandb_project_id: str = "mantra-dev-run-3") -> None:
        full_history = setup_wandb(wandb_project_id=wandb_project_id)
        df = convert_history(full_history)
        df = process_df(df)
        self.df = df

    def get(self):
        return self.df

    def get_task_means(self) -> Tuple[List[float], List[float]]:
        categories = [
            "betti_0",
            "betti_1",
            "betti_2",
            "name",
            "orientability",
        ]  # categories = self.df.to_numpy()
        values = []
        errors = []

        for c in categories:
            indeces = self.df["Task"] == c
            filtered_df = self.df.iloc[indeces.to_numpy()]
            mean_accuracy = filtered_df["Mean Accuracy"].mean()
            std_accuracy = filtered_df["Std Accuracy"].mean()
            mean_train_accuracy = filtered_df["Mean Train Accuracy"].mean()
            std_train_accuracy = filtered_df["Std Train Accuracy"].mean()
            values.append(mean_accuracy)
            errors.append(std_accuracy)

        return values, errors

    def save(self, path: str):
        raise NotImplementedError()

    def load(self, path: str):
        raise NotImplementedError()
