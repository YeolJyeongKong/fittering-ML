# FROM sinjy1203/fittering-measurements:2.0
# FROM jenkins/jenkins:jdk11
# # FROM public.ecr.aws/lambda/python:3.8
# # COPY requirements.txt ./
# # RUN apt update && apt-get install python3 python3-pip -y
# # RUN pip install -r requirements.txt
# RUN apt-get update && apt-get -y install sudo
# RUN sudo apt-get install wget
# RUN wget https://www.python.org/ftp/python/3.8.5/Python-3.8.5.tar.xz
# RUN tar xvf Python-3.8.5.tar.xz
# WORKDIR /Python-3.8.5
# RUN apt -y install build-essential
# RUN apt-get install zlib1g-dev
# RUN ./configure
# RUN make
# RUN make altinstall

# FROM jenkins/jenkins:latest
# USER root
# RUN apt-get update
# RUN apt-get install -y python-pip
# # Install app dependencies
# RUN pip install --upgrade pip

# Dockerfile
# FROM jenkins:latest
# USER root

# RUN apt-get update

# # pip 설치
# RUN apt-get install -y python-pip
# ENV JAVA_ARGS -Xms512m -Xmx1024m

# RUN apt-get update && \
#     apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget


# FROM jenkins:latest
FROM jenkins/jenkins:latest
USER root
COPY requirements.txt ./
RUN apt update && apt install -y python3 python3-pip
RUN pip install -r requirements.txt --break-system-packages
# RUN apt-get install -y python-pip
# # Install app dependencies
# RUN pip install --upgrade pip

# COPY ./ ${LAMBDA_TASK_ROOT}
# RUN ls -la ${LAMBDA_TASK_ROOT}/*
# COPY requirements.serving.txt ${LAMBDA_TASK_ROOT}

# Install the specified packages
# RUN pip install torch==1.13.1+cpu torchvision==0.14.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
# RUN pip install -r requirements.serving.txt
# RUN yum update && yum install -y libglvnd-glx

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
# CMD [ "lambda_function.handler" ]