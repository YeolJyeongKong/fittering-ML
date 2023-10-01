resource "aws_iam_role" "ecr_s3_full_access_role" {
  name = "ecr_s3_full_access_role"
  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": ["ec2.amazonaws.com"]
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

}

resource "aws_iam_policy_attachment" "s3_full_access_attach" {
  name       = "s3_full_access_attach"
  roles      = [aws_iam_role.ecr_s3_full_access_role.name]
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_policy_attachment" "ecr_full_access_attach" {
  name       = "ecr_full_access_attach"
  roles      = [aws_iam_role.ecr_s3_full_access_role.name]
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess"
}


resource "aws_iam_instance_profile" "ecr_s3_full_access_profile" {
  name = "ecr_s3_full_access_profile"
  role = aws_iam_role.ecr_s3_full_access_role.name
}

