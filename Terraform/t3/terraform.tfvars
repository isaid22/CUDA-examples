# terraform.tfvars

key_name           = "kt6238key"
tags_name           = "t2s"
subnet_id          = "subnet-02140b7ce4d2ad408"
security_group_id  = "sg-00ee8b7a416fa0d47"
ami_id             = "ami-0f9de6e2d2f067fca"  # Pubntu server 22.04 LTS
instance_type      = "t2.small"             # Provide the EC2 instance type here
docker_image       = "apache/kafka"  # Specify the Docker image here
install_dir        = "/home/ubuntu" 
