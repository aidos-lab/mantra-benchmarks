#!/bin/bash

python ./experiments/run.py \
    --mode "single" \
    --config "./configs/gat_betti_numbers_degree_transform_onehot.yaml" \
    --wandb "mantra-proj" \
    --checkpoints "./checkpoints"
