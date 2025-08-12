# terraform.tfvars

key_name           = "kt6238key"
tags_name           = "t2large"
subnet_id          = "subnet-02140b7ce4d2ad408"
security_group_id  = "sg-00ee8b7a416fa0d47"
ami_id             = "ami-020cba7c55df1f615"  # Ubuntu 24.04 LTS
instance_type      = "t2.large"             # Provide the EC2 instance type here
docker_image       = "apache/kafka"  # Specify the Docker image here
install_dir        = "/home/ubuntu" 
