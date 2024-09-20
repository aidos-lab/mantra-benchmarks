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
from datasets.dataset_types import DatasetType


def verify_config(
    config: ConfigExperimentRun, number_of_barycentric_subdivisions: int
):
    """
    Verifies that it is intended to run the configuration.
    Returns false if the dataset type is != NO_NAMELESS_2D and the number
    of barycentric subdivisions is > 0.
    """
    is_verified = not (
        number_of_barycentric_subdivisions > 0
        and config.ds_type != DatasetType.NO_NAMELESS_2D
    )
    if not is_verified:
        print(
            f"[INFO] {config.ds_type} dataset type is not intended to run with {number_of_barycentric_subdivisions} number of barycentric subdivisions. Thus, skipping evaluation."
        )  # noqa

    bary_too_large = number_of_barycentric_subdivisions > 1
    if bary_too_large:
        print(
            "[INFO] Number of barycentric subdivisions can currently not be > 1."
        )

    return is_verified and not bary_too_large


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

    if not verify_config(config, number_of_barycentric_subdivisions):
        return

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
    results.load(".ignore_temp")

    # get the benchmarks:
    for file in files:

        # load config
        config_file = os.path.join(config_dir, file)
        config = load_config(config_file)

        if not verify_config(config, number_of_barycentric_subdivisions):
            continue

        n_existing = results.exists(config, number_of_barycentric_subdivisions)
        if n_existing == n_runs:
            print(
                f"[INFO] Skipping testing {config} with n_bary_subdv {number_of_barycentric_subdivisions} because sufficient existing entries were found."
            )
            continue

        for run in range(n_existing, n_runs):

            # load weights
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
    barycentric_subdivisions: int = args.barycentric_subdivisions

    if args_dict["mode"] == "single":
        config = load_config(args_dict["config"])
        checkpoint_path = config.get_checkpoint_path(
            args_dict["checkpoints"], run=run
        )
        test(
            config=config,
            checkpoint_path=checkpoint_path,
            data_dir=data_dir,
            number_of_barycentric_subdivisions=barycentric_subdivisions,
            devices=devices,
        )
    elif args_dict["mode"] == "all":
        test_all(
            checkpoint_dir=args_dict["checkpoints"],
            config_dir=args_dict["Configs"],
            data_dir=data_dir,
            number_of_barycentric_subdivisions=barycentric_subdivisions,
            devices=devices,
        )
    else:
        ValueError("Unknown mode")
