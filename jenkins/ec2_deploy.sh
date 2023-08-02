DOCKER_IMAGE=210651441624.dkr.ecr.ap-northeast-2.amazonaws.com/human_size_predict
aws ecr get-login-password --region ap-northeast-2 | docker login --username AWS --password-stdin 210651441624.dkr.ecr.ap-northeast-2.amazonaws.com
docker pull $DOCKER_IMAGE
docker stop $(docker ps -aq)
docker ps -q --filter ancestor=$DOCKER_IMAGE | xargs -r docker stop
docker run -i --rm -p 80:3000 $DOCKER_IMAGE serve