from experiments.configs import load_config
from experiments.run_experiment import run_configuration
import os

def run_configs_folder():
    config_dir = "./configs"
    files = os.listdir(config_dir)
    for file in files:
        for _ in range(5):
            config_file = os.path.join(config_dir, file)
            config = load_config(config_file)
            run_configuration(config)


if __name__ == "__main__":
    run_configs_folder()
