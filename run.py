from experiments.configs import load_config, ConfigExperimentRun
from experiments.run_experiment import run_configuration
import os
import argparse
from typing import Dict, Any, Optional


def print_info(config: ConfigExperimentRun):
    print("[INFO] Using configuration:", config)


def run_configs_folder(
    args_dict: Dict[str, Any], checkpoint_folder: Optional[str] = None
):
    config_dir = "./configs"
    files = os.listdir(config_dir)
    for file in files:
        for _ in range(5):
            config_file = os.path.join(config_dir, file)
            config = load_config(config_file)
            print("[INFO] Using configuration file:", config_file)
            print_info(config)

            checkpoint_path = None
            if checkpoint_folder:
                checkpoint_path = config.get_checkpoint_path(checkpoint_folder)
                print("[INFO] Using checkpoint folder:", checkpoint_path)
            else:
                print("[INFO] No checkpoint folder.")

            config.logging.wandb_project_id = args_dict["wandb"]
            run_configuration(config, save_checkpoint_path=checkpoint_path)


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
        default=None,
        help="Path where the model checkpoints are stored.",
    )
    parser.add_argument(
        "--wandb", type=str, default="mantra-dev", help="Wandb project id."
    )

    args = parser.parse_args()
    args_dict = vars(args)

    if args_dict["mode"] == "all":
        run_configs_folder(
            args_dict, checkpoint_folder=args_dict["checkpoints"]
        )
        exit(0)

    if args_dict["mode"] == "single":
        config = load_config(args_dict["config"])
        config.logging.wandb_project_id = args_dict["wandb"]
        print_info(config)
        config_path = None
        if args_dict["checkpoints"]:
            config_path = config.get_checkpoint_path(args_dict["checkpoints"])
        run_configuration(config, save_checkpoint_path=config_path)
        exit(0)
