#!/bin/bash

sudo /deps/venv/bin/python ./experiments/train.py \
    --mode "all" \
    --Configs "/data/configs/" \
    --wandb "mantra-proj-transformer-3manifolds" \
    --checkpoints "/data/checkpoints" \
    --data "/data"
