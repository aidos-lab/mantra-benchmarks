import sys
import os

sys.path.append(os.curdir)
from experiments.utils.configs import load_config, ConfigExperimentRun
from experiments.utils.run_experiment import (
    benchmark_configuration,
)
import os
import argparse
from experiments.utils.result_collection import ResultCollection
from typing import List


def test(
    config: ConfigExperimentRun,
    checkpoint_path: str,
    data_dir: str = "./data",
    number_of_barycentric_subdivisions: int = 0,
    devices=None,
):
    """
    Runs the benchmark for one specific configuration and trained weights.
    """

    print("[INFO] Testing with config", config)
    print("[INFO] Testing with checkpoint path:", checkpoint_path)
    print("[INFO] Testing with data path:", data_dir)

    benchmark_configuration(
        config=config,
        save_checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        number_of_barycentric_subdivisions=number_of_barycentric_subdivisions,
        devices=devices,
    )


def test_all(
    checkpoint_dir: str,
    config_dir: str = "./configs",
    n_runs=5,
    data_dir: str = "./data",
    number_of_barycentric_subdivisions: int = 0,
    devices=None,
):
    """
    Tests all configurations in a config directory. Assumes that for every run and every config there is a
    corresponding checkpoint in checkpoint_dir given.
    """

    files = os.listdir(config_dir)
    results = ResultCollection()

    # get the benchmarks:
    for file in files:
        for run in range(n_runs):

            # load config and weights
            config_file = os.path.join(config_dir, file)
            config = load_config(config_file)
            checkpoint_path = config.get_checkpoint_path(
                checkpoint_dir, run=run
            )

            # run benchmark
            out = benchmark_configuration(
                config=config,
                save_checkpoint_path=checkpoint_path,
                data_dir=data_dir,
                number_of_barycentric_subdivisions=number_of_barycentric_subdivisions,
                devices=devices,
            )

            # each metric is repeated for every barycentric subdivision
            for idx in range(number_of_barycentric_subdivisions + 1):
                results.add(
                    data=out[idx],
                    config=config,
                    barycentric_subdivision_idx=idx,
                )

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
        help="Path to .yaml configuration for experiment if running 'single' mode. ",
    )
    parser.add_argument(
        "--Configs",
        type=str,
        default="./configs",
        help="Path to folder containing all configurations if running 'all' mode.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="./data",
        help="Folder where data is stored in.",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        help="Path where the model checkpoints are stored.",
    )
    parser.add_argument(
        "--barycentric_subdivisions",
        type=int,
        help="Maximum number of barycentric subdivisions to perform for test evaluation.",
        default=3,
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        type=int,
        help="List of GPU IDs to use.",
        default=[0],
    )
    parser.add_argument(
        "--run",
        type=int,
        help="If testing a single checkpoint, specifies which run to take.",
        default=0,
    )

    args = parser.parse_args()
    args_dict = vars(args)
    data_dir = args.data
    devices: List[int] = args.devices
    run: int = args.run

    if args_dict["mode"] == "single":
        config = load_config(args_dict["config"])
        checkpoint_path = config.get_checkpoint_path(
            args_dict["checkpoints"], run=run
        )
        test(
            config=config,
            checkpoint_path=checkpoint_path,
            data_dir=data_dir,
            number_of_barycentric_subdivisions=args_dict[
                "barycentric_subdivisions"
            ],
            devices=devices,
        )
    elif args_dict["mode"] == "all":
        test_all(
            checkpoint_dir=args_dict["checkpoints"],
            config_dir=args_dict["Configs"],
            data_dir=data_dir,
            number_of_barycentric_subdivisions=args_dict[
                "barycentric_subdivisions"
            ],
            devices=devices,
        )
    else:
        ValueError("Unknown mode")
