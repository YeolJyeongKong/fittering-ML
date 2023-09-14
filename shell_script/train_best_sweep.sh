#!/bin/bash

echo "경로를 입력하세요: "
read path

python src/train.py --config-path $path/.hydra --config-name config trainer.product_encode.max_epochs=10