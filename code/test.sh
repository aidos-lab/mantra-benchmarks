#!/bin/bash

python ./experiments/test.py \
    --mode "single" \
    --config "./configs/gat_betti_numbers_degree_transform_onehot.yaml" \
    --checkpoints "../data/checkpoints" \
    --data "../data"
