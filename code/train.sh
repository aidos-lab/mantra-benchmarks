#!/bin/bash

sudo /deps/venv/bin/python ./experiments/train.py \
    --mode "all" \
    --Configs "/data/configs_3manifolds_random_transform" \
    --wandb "mantra-proj-3manifolds-camera-ready-random-transforms" \
    --checkpoints "/data/checkpoints/camera-ready-random-transforms" \
    --data "/data" \
    --devices 0 2
