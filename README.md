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

- see [train.sh](./train.sh), [test.sh](./test.sh), [generate_configs.sh](./generate_configs.sh)

1. Generate experiment configurations: `python generate_configs.py`
2. Run experiments:

```s
$ python ./experiments/run.py --mode "single" --config "/path/to/config.yaml" --wandb "wandb-project-id"
```
for running a single experiment or
```s
$ python ./experiments/run.py --mode "all" --wandb "wandb-project-id"
```
for running all experiments.

3. Run benchmarking

```s
$ python ./experiments/run.py --mode "<single/all>" --checkpoints "<checkpoints/to/be/benchmarked>"
```

4. Result processing and table generation.

```s
$ python ./experiments/generate_tables.py
# optional
$ pandoc results.md -o results.pdf -V geometry:margin=0.1in -V geometry:a3paper
```


## Results

Results can be inspected in [notebooks/interpret_results.ipynb](./notebooks/interpret_results.ipynb)

## Development

- Formatting using black 24.4.2. Install via pip (`pip install black==24.4.2`).


## Dataset distribution

| Name              |           |        |                 |       |      |     |
| ----------------- | --------- | ------ | --------------- | ----- | ---- | --- |
| 'S^2'             |  'RP^2'   | 'T^2'  | 'Klein bottle'  | ''    |      |     |
| 306               | 1367      | 2229   | 4655            | 34584 |      |     |
|                   |           |        |                 |       |      |     |
| **Orientability** |           |        |                 |       |      |     |
| True              | False     |        |                 |       |      |     |
| 3420              | 39718     |        |                 |       |      |     |
|                   |           |        |                 |       |      |     |
| **Betti Numbers** |           |        |                 |       |      |     |
| **Betti_0**       |           |        |                 |       |      |     |
| 1                 |           |        |                 |       |      |     |
| 43138             |           |        |                 |       |      |     |
| **Betti_1**       |           |        |                 |       |      |     |
| 0                 | 1         | 2      | 3               | 4     | 5    | 6   |
| 1670              | 4655      | 14146  | 13694           | 7917  | 1022 | 34  |
| **Betti_2**       |           |        |                 |       |      |     |
| 0                 | 1         |        |                 |       |      |     |
| 39718             | 3420      |        |                 |       |      |     |

