#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fittering-measurements

export WANDB_DIR=./wandb_sweep

wandb sweep ./configs/sweep_human_size.yaml > temp_output.txt 2>&1

output_path="temp_output.txt"
output=$(<"$output_path")
export SWEEP_ID=$(echo "$output" | awk '/Creating sweep with ID:/ {print $6}')
rm temp_output.txt

wandb agent sinjy1203/human_size_sweep/$SWEEP_ID

python shell_script/sweep/sweep_human_size_best.py
