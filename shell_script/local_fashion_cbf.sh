#!/bin/bash
read -p "relative output dir: " OUTPUT_DIR

# 입력이 없으면 스크립트를 종료
if [ -z "$OUTPUT_DIR" ]; then
  echo "입력된 값이 없으므로 스크립트를 종료합니다."
  exit 1
fi

export OUTPUT_DIR
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fashion-cbf-env
BENTOML_CONFIG=./configs/bentoml_config.yaml bentoml serve serving.bentoml.service_fashion_cbf:svc --development