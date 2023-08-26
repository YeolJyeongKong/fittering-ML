#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fashion-ubf-env
bentoml serve serving.bentoml.service_fashion_ubf:svc --development --reload