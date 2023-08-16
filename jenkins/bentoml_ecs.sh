#!/bin/bash
read -p "relative output dir: " OUTPUT_DIR

# 입력이 없으면 스크립트를 종료
if [ -z "$OUTPUT_DIR" ]; then
  echo "입력된 값이 없으므로 스크립트를 종료합니다."
  exit 1
fi


export OUTPUT_DIR
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fittering-measurements-cpu
bentoml build -f $OUTPUT_DIR/bentofile.yaml
bentoml containerize human_size_predict:latest -t 210651441624.dkr.ecr.ap-northeast-2.amazonaws.com/human_size_predict:latest

aws ecr get-login-password --region ap-northeast-2 | docker login --username AWS --password-stdin 210651441624.dkr.ecr.ap-northeast-2.amazonaws.com
docker push 210651441624.dkr.ecr.ap-northeast-2.amazonaws.com/human_size_predict:latest


# ecs service update
CLUSTER_NAME=human-size-predict-cluster
SERVICE_NAME=human-size-predict-cluster
aws ecs update-service --cluster ${CLUSTER_NAME} --service ${SERVICE_NAME} --force-new-deployment