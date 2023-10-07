#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fashion-cbf-env
export AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key)
export AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id)
python ./shell_script/container_test/bentofile_local/add_dockertemplate.py --bentofile_name bentofile_product_encode.yaml
bentoml build -f ./shell_script/container_test/bentofile_local/bentofile_product_encode.yaml
bentoml containerize --build-arg AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
                    --build-arg AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
                    fashion-cbf:latest \
                    -t 210651441624.dkr.ecr.ap-northeast-2.amazonaws.com/fashion-cbf:latest
sudo docker run --rm -p 3000:3000 210651441624.dkr.ecr.ap-northeast-2.amazonaws.com/fashion-cbf:latest