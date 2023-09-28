resource "aws_iam_role" "ecr_fully_access_role" {
  name = "ecr_fully_access_role"
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

resource "aws_iam_policy" "ecr_fully_access_policy" {
  name = "ecr_fully_access_policy"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "ecr:*",
        ]
        Effect   = "Allow"
        Resource = "*"
      },
    ]
  })
}

resource "aws_iam_policy_attachment" "ecr_fully_access_attach" {
  name       = "ecr_fully_access_attach"
  roles      = [aws_iam_role.ecr_fully_access_role.name]
  policy_arn = aws_iam_policy.ecr_fully_access_policy.arn
}

resource "aws_iam_instance_profile" "ecr_fully_access_profile" {
  name = "ecr_fully_access_profile"
  role = aws_iam_role.ecr_fully_access_role.name
}

