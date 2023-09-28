resource "aws_key_pair" "ec2_jenkins" {
    key_name = "ec2_jenkins"
    public_key = file("~/Documents/aws/keypairs/ec2_jenkins.pub")
}