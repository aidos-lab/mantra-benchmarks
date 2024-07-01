from experiments.configs import load_config, ConfigExperimentRun
from experiments.run_experiment import (
    benchmark_configuration,
)
import os
import argparse
from typing import Dict, List
from metrics.tasks import TaskType
import pandas as pd


def test(config: ConfigExperimentRun, checkpoint_path: str):
    print("[INFO] Testing with config", config)
    print("[INFO] Testing with checkpoint path:", checkpoint_path)

    benchmark_configuration(
        config=config, save_checkpoint_path=checkpoint_path
    )


class Result:

    def __init__(
        self, data: Dict[str, float], config: ConfigExperimentRun
    ) -> None:
        self.data = data
        self.config = config


class ResultCollection:
    collection: Dict[TaskType, List[Result]]

    def __init__(self) -> None:
        self.collection = {
            TaskType.BETTI_NUMBERS: [],
            TaskType.NAME: [],
            TaskType.ORIENTABILITY: [],
        }

    def add(self, data: Dict[str, float], config: ConfigExperimentRun):
        self.collection[config.task_type].append(
            Result(data=data, config=config)
        )

    def save(self, t_file_prefix: str = "res"):

        for task_type in TaskType:
            t_file = f"{t_file_prefix}_{task_type.name.lower()}.csv"

            data = []
            for x in self.collection[task_type]:
                result = x.data
                result["type_model"] = x.config.conf_model.type.name.lower()
                result["transform"] = x.config.transforms.name.lower()
                data.append(x.data)

            df = pd.DataFrame(data)
            df.to_csv(t_file, index=False)


def test_all(checkpoint_dir: str, config_dir: str = "./configs"):
    files = os.listdir(config_dir)

    results = ResultCollection()

    # get the benchmarks:
    for file in files:
        config_file = os.path.join(config_dir, file)
        config = load_config(config_file)
        checkpoint_path = config.get_checkpoint_path(checkpoint_dir)
        out = benchmark_configuration(
            config=config, save_checkpoint_path=checkpoint_path
        )

        results.add(data=out[0], config=config)

        results.save(".ignore_temp")

    results.save("results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Argument parser for experiment configurations."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        help="'all' for running all configurations in the ./configs folder, or 'single' for running a single model.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to .yaml configuration for experiment if running 'single mode.",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        help="Path where the model checkpoints are stored.",
    )

    args = parser.parse_args()
    args_dict = vars(args)

    if args_dict["mode"] == "single":
        config = load_config(args_dict["config"])
        checkpoint_path = config.get_checkpoint_path(args_dict["checkpoints"])
        test(config=config, checkpoint_path=checkpoint_path)
    elif args_dict["mode"] == "all":
        test_all(checkpoint_dir=args_dict["checkpoints"])
    else:
        ValueError("Unknown mode")
