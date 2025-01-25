#!/bin/bash

sudo /deps/venv/bin/python ./experiments/generate_configs.py \
    --max_epochs 20 \
    --lr 0.001 \
    --config_dir "/data/configs" \
