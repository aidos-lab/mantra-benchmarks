#!/bin/bash

python ./experiments/train.py \
    --mode "single" \
    --config "./configs/gat_betti_numbers_degree_transform_onehot.yaml" \
    --wandb "mantra-proj" \
    --checkpoints "../data/checkpoints" \
    --data "../data"
