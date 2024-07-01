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


def test(config: ConfigExperimentRun, checkpoint_path: str):
    """
    Runs the benchmark for one specific configuration and trained weights.
    """

    print("[INFO] Testing with config", config)
    print("[INFO] Testing with checkpoint path:", checkpoint_path)

    benchmark_configuration(
        config=config, save_checkpoint_path=checkpoint_path
    )


def test_all(checkpoint_dir: str, config_dir: str = "./configs", n_runs=5):
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
                config=config, save_checkpoint_path=checkpoint_path
            )

            # add benchmarking results
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
