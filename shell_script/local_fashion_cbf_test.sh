#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fashion-cbf-env
BENTOML_CONFIG=./configs/bentoml_config.yaml bentoml serve serving.bentoml.service_fashion_cbf:svc --development --reload