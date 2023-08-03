DOCKER_IMAGE=210651441624.dkr.ecr.ap-northeast-2.amazonaws.com/human_size_predict
aws ecr get-login-password --region ap-northeast-2 | docker login --username AWS --password-stdin 210651441624.dkr.ecr.ap-northeast-2.amazonaws.com

# Docker 이미지 개수 확인
image_count=$(docker images -q | wc -l)

# 이미지 개수가 3개 이상인지 확인
if [ "$image_count" -ge 3 ]; then
  # 모든 Docker 이미지 삭제
  docker rmi $(docker images -q)
  echo "모든 Docker 이미지를 삭제했습니다."
else
  echo "Docker 이미지가 ${image_count}개 이므로 삭제하지 않습니다."
fi

docker pull $DOCKER_IMAGE
docker stop $(docker ps -aq)
docker ps -q --filter ancestor=$DOCKER_IMAGE | xargs -r docker stop
docker run -i --rm -p 80:3000 $DOCKER_IMAGE serve