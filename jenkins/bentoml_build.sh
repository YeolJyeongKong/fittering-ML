#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fittering-measurements-cpu
bentoml build -f /home/shin/VScodeProjects/fittering-ML/outputs/2023-08-02/00-17-34/bentofile.yaml
bentoml containerize human_size_predict:latest