#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate human-size-env
bentoml serve serving.bentoml.service_human_size:svc --development --reload