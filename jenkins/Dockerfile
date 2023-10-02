FROM jenkins/jenkins:jdk11
USER root
# COPY requirements.txt ./
RUN apt-get update
RUN rm /var/lib/dpkg/info/python*
RUN apt-get install -y python3 python3-pip
# RUN rm /var/lib/dpkg/info/python*
# RUN dpkg --configure -a
# RUN apt update -y


