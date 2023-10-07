#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate human-size-env
bentoml build -f bentofile/bentofile_human_size.yaml
bentoml containerize human_size_predict:latest -t 210651441624.dkr.ecr.ap-northeast-2.amazonaws.com/human_size_predict:latest

aws ecr get-login-password --region ap-northeast-2 | docker login --username AWS --password-stdin 210651441624.dkr.ecr.ap-northeast-2.amazonaws.com
docker push 210651441624.dkr.ecr.ap-northeast-2.amazonaws.com/human_size_predict:latest


# ecs service update
CLUSTER_NAME=ML-cluster
SERVICE_NAME=human-size-service
aws ecs update-service --cluster ${CLUSTER_NAME} --service ${SERVICE_NAME} --force-new-deployment