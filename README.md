# MANTRA: The Manifold Triangulations Assemblage — Benchmarks Repository

![GitHub contributors](https://img.shields.io/github/contributors/aidos-lab/mantra-benchmarks) ![License](https://img.shields.io/github/license/aidos-lab/mantra-benchmarks) 

This is the benchmarks repository accompanying our [ICLR 2025 paper](https://openreview.net/pdf?id=X6y5CC44HM) on manifold triangulations. If you are looking for the dataset itself and the data loader, check out [the data repository](https://github.com/aidos-lab/MANTRA).

Please use the following citation for our work:

```bibtex
@inproceedings{Ballester25a,
  title         = {{MANTRA}: {T}he {M}anifold {T}riangulations {A}ssemblage},
  author        = {Rubén Ballester and Ernst Röell and Daniel Bīn Schmid and Mathieu Alain and Sergio Escalera and Carles Casacuberta and Bastian Rieck},
  year          = 2025,
  booktitle     = {International Conference on Learning Representations},
  url           = {https://openreview.net/forum?id=X6y5CC44HM},
}
```

## Setup

Either via Docker or a standard setup. Start with:

```s
git submodule update --init --recursive
```

### Standard setup

1. Setup python version 3.10.13. E.g. with `pyenv install 3.10.13` and `pyenv local 3.10.13`. Set up a virtual environment, e.g. via `python3 -m venv ./venv` and `source ./venv/bin/activate`,
2. Install dependencies via poetry
```s
$ python -m pip install --upgrade pip
$ pip install poetry
$ poetry install
```
3. `pip install -e ./dependencies/TopoModelX/`

### Docker

Depending on your setup, you may need to run the docker commands via `sudo`. 

0. Install Docker on your device. On Ubuntu, for instance, refer to [official_installation_instructions](./https://docs.docker.com/engine/install/ubuntu/)
1. Edit the variables `USER_NAME` to `GROUP_ID` in the [Dockerfile](./containerization/Dockerfile) 
2. `docker compose up --build -d`. Check that container is running via `docker ps`
3. `docker exec -it mantra_container /bin/bash`
4. When inside the container, run `source /deps/venv/bin/activate` to source the ready virtual environment

## Usage

- see [train.sh](./train.sh), [test.sh](./test.sh), [generate_configs.sh](./generate_configs.sh)

0. Change in code directory.
1. Generate experiment configurations: `./generate_configs.sh`
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
$ pandoc results_2fullmanifolds.md -o results_2fullmanifolds.pdf -V geometry:margin=0.1in -V geometry:a1paper
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

