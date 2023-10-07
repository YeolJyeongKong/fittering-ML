#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fashion-cbf-env
bentoml build -f bentofile/bentofile_product_encode.yaml
bentoml export fashion-cbf:latest s3://fittering-bento/fashion-cbf.bento
