#!/bin/bash
export WANDB_DIR=./wandb_sweep

wandb sweep ./configs/sweep_product_encode.yaml > temp_output.txt 2>&1

output_path="temp_output.txt"
output=$(<"$output_path")
export SWEEP_ID=$(echo "$output" | awk '/Creating sweep with ID:/ {print $6}')
rm temp_output.txt

wandb agent sinjy1203/product_encode_sweep/$SWEEP_ID

python shell_script/sweep/sweep_product_encode_best.py

