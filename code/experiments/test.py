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


def test(
    config: ConfigExperimentRun,
    checkpoint_path: str,
    data_dir: str = "./data",
    number_of_barycentric_subdivisions: int = 0,
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
    )


def test_all(
    checkpoint_dir: str,
    config_dir: str = "./configs",
    n_runs=5,
    data_dir: str = "./data",
    number_of_barycentric_subdivisions: int = 0,
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
            )

            # Modify output metric keys to contemplate the number of barycentric subdivisions
            out_processed = (
                dict()
            )  # The dict will contain the processed metrics for all barycentric subdivisions.
            # We assume that we always use one test dataloader when testing.
            for idx in range(
                number_of_barycentric_subdivisions + 1
            ):  # Each metric is repeated for each
                # barycentric subdivision
                for key in out[idx][0].keys():
                    out_processed[f"{key}_bs_{idx}"] = out[idx][0][key]

            # add benchmarking results
            results.add(data=out_processed, config=config)
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

    args = parser.parse_args()
    args_dict = vars(args)
    data_dir = args.data

    if args_dict["mode"] == "single":
        config = load_config(args_dict["config"])
        checkpoint_path = config.get_checkpoint_path(args_dict["checkpoints"])
        test(
            config=config,
            checkpoint_path=checkpoint_path,
            data_dir=data_dir,
            number_of_barycentric_subdivisions=args_dict[
                "barycentric_subdivisions"
            ],
        )
    elif args_dict["mode"] == "all":
        test_all(
            checkpoint_dir=args_dict["checkpoints"],
            config_dir=args_dict["Configs"],
            data_dir=data_dir,
            number_of_barycentric_subdivisions=args_dict[
                "barycentric_subdivisions"
            ],
        )
    else:
        ValueError("Unknown mode")
