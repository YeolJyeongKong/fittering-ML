#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fittering-measurements-cpu
bentoml build -f $1/bentofile.yaml
bentoml containerize human_size_predict:latest -t 210651441624.dkr.ecr.ap-northeast-2.amazonaws.com/human_size_predict:latest

aws ecr get-login-password --region ap-northeast-2 | docker login --username AWS --password-stdin 210651441624.dkr.ecr.ap-northeast-2.amazonaws.com
docker push 210651441624.dkr.ecr.ap-northeast-2.amazonaws.com/human_size_predict:latest

KEYPAIR=/home/shin/Documents/aws/keypairs/ubuntu-desktop-keypairs.pem
REMOTE_HOST=ec2-54-180-148-94.ap-northeast-2.compute.amazonaws.com
scp -i $KEYPAIR ./jenkins/ec2_deploy.sh ec2-user@$REMOTE_HOST:~/
ssh -i $KEYPAIR ec2-user@$REMOTE_HOST "chmod +x ec2_deploy.sh"
ssh -i $KEYPAIR ec2-user@$REMOTE_HOST ./ec2_deploy.sh