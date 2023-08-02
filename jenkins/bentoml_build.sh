#!/bin/bash
# ANACONDA_DIR="/home/shin/anaconda3"  # Anaconda가 설치된 디렉토리 경로
# ENV_NAME="fittering-measurements-cpu"         # 가상환경 이름

# # Anaconda가 설치된 디렉토리로 이동합니다.
# cd $ANACONDA_DIR

# # Anaconda를 활성화하는 스크립트를 실행합니다.
# source bin/activate

# # 원하는 가상환경을 활성화합니다.
# conda activate $ENV_NAME



# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate fittering-measurements-cpu
# bentoml build -f /home/shin/VScodeProjects/fittering-ML/outputs/2023-08-02/00-17-34/bentofile.yaml
# bentoml containerize human_size_predict:latest

echo $PATH
echo $HOME
/home/shin/anaconda3/bin/activate fittering-measurements-cpu
chmod ug+x /app/local/anaconda3/bin/activate
bentoml build -f /home/shin/VScodeProjects/fittering-ML/outputs/2023-08-02/00-17-34/bentofile.yaml
