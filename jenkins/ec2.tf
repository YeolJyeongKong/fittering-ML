resource "aws_instance" "jenkins" {
    ami = "ami-02288bc8778f3166f"
    instance_type = "t2.micro"
    key_name = aws_key_pair.ec2_jenkins.key_name
    vpc_security_group_ids = [
        aws_security_group.ssh_jenkins.id
    ]
    iam_instance_profile = aws_iam_instance_profile.ecr_s3_full_access_profile.name
    tags = {
        Name = "jenkins"
    }
    user_data = file("./jenkins_install.sh")
}