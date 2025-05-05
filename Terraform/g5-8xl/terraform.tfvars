# terraform.tfvars

key_name           = "
tags_name           = "g5-tf-param"
subnet_id          = "subnet-<YOUR SUBNET ID>"
security_group_id  = "sg-<YOUR SECURITY GROUP ID>"
ami_id             = "ami-0284440fc8de0d5e8"  # Provide the AMI ID here
instance_type      = "g5.8xlarge"             # Provide the EC2 instance type here
docker_image       = "rickblaine/vllm-client:g512xl"  # Specify the Docker image here

