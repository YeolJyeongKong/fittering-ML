FROM jenkins/jenkins:lts
USER root
# COPY requirements.txt ./
RUN apt-get update -y
RUN apt-get install -y python3 python3-pip

# install docker
RUN apt-get install ca-certificates curl gnupg sudo && \
    sudo install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    sudo chmod a+r /etc/apt/keyrings/docker.gpg && \
    echo \
        "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
        "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    sudo apt-get update && \
    apt-get install -y docker-ce && \
    sudo usermod -aG docker jenkins

# RUN rm /var/lib/dpkg/info/python*
# RUN dpkg --configure -a
# RUN apt update -y


