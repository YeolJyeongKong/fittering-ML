#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate human-size-env
bentoml build -f bentofile/bentofile_human_size.yaml
bentoml export human_size_predict:latest s3://fittering-bento/human-size-predict.bento
