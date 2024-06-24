from experiments.configs import load_config, ConfigExperimentRun
from experiments.run_experiment import (
    run_configuration,
    benchmark_configuration,
)
import os
import argparse
from typing import Dict, Any, Optional


def test(config: ConfigExperimentRun, checkpoint_path: str):
    print("[INFO] Testing with config", config)
    print("[INFO] Testing with checkpoint path:", checkpoint_path)

    benchmark_configuration(
        config=config, save_checkpoint_path=checkpoint_path
    )


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

    config = load_config(args_dict["config"])
    checkpoint_path = config.get_checkpoint_path(args_dict["checkpoints"])
    test(config=config, checkpoint_path=checkpoint_path)
