## jenkins docker run
sudo docker run -itd -p 8080:8080 -v /var/run/docker.sock:/var/run/docker.sock --name jenkins-server jenkins-server