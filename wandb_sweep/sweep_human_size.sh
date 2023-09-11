#!/bin/bash
export WANDB_DIR=./wandb_sweep/
  
wandb sweep wandb_sweep/sweep_human_size.yaml > temp_output.txt 2>&1

last_line=$(tail -n 1 temp_output.txt)
rm temp_output.txt
RUN=$(echo "$last_line" | awk -F':' '{print $NF}')
$RUN
