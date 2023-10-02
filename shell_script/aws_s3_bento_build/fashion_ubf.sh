#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fashion-ubf-env
bentoml build -f ./bentofile/bentofile_ubf.yaml
bentoml export fashion-ubf:latest s3://fittering-bento/fashion-ubf.bento
