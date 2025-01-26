#!/bin/bash

python ./experiments/generate_configs.py \
    --max_epochs 10 \
    --lr 0.001 \
    --config_dir "/data/configs_3manifolds_random_transform" \
    --three_manifold_only \
    --random_transform_only \
    # --degree_transform_only
