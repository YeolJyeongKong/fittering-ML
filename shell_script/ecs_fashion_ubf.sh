#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fashion-ubf-env
bentoml build -f ./serving/bentoml/bentofile_ubf.yaml
bentoml containerize fashion-ubf:latest -t 210651441624.dkr.ecr.ap-northeast-2.amazonaws.com/fashion-ubf:latest

aws ecr get-login-password --region ap-northeast-2 | docker login --username AWS --password-stdin 210651441624.dkr.ecr.ap-northeast-2.amazonaws.com
docker push 210651441624.dkr.ecr.ap-northeast-2.amazonaws.com/fashion-ubf:latest


# ecs service update
CLUSTER_NAME=ML-cluster
SERVICE_NAME=fashion-ubf-service
aws ecs update-service --cluster ${CLUSTER_NAME} --service ${SERVICE_NAME} --force-new-deployment