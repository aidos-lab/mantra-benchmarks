# MANTRA: Manifold Triangulations

## Setup

0. Clone with submodules. After cloning, run `git submodule update --init --recursive`
1. Setup python version 3.10.13. E.g. with `pyenv install 3.10.13` and `pyenv local 3.10.13`. Set up a virtual environment, e.g. via `python3 -m venv ./venv` and `source ./venv/bin/activate`,
2. Install dependencies via poetry
```s
$ pip install --upgrade pip
$ pip install poetry
$ poetry install
```
3. Install toponetx dependency
4. `pip install -e ./mantra/`

## Usage

1. Generate experiment configurations: `python generate_configs.py`
2. Run experiments:

```s
$ python ./run.py --mode "single" --config "/path/to/config.yaml" --wandb "wandb-project-id"
```
for running a single experiment or
```s
$ python ./run.py --mode "all"
```
for running all experiments.

## Results

Results can be inspected in [notebooks/export_results.ipynb](./notebooks/export_results.ipynb)

## Development

- Formatting using black 24.4.2. Install via pip (`pip install black==24.4.2`).
