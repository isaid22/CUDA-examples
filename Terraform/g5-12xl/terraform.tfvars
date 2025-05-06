# terraform.tfvars

key_name           = "kt6238key"
tags_name           = "g5tf"
subnet_id          = "subnet-02140b7ce4d2ad408"
security_group_id  = "sg-00ee8b7a416fa0d47"
ami_id             = "ami-0284440fc8de0d5e8"  # Provide the AMI ID here
instance_type      = "g5.12xlarge"             # Provide the EC2 instance type here
docker_image       = "rickblaine/vllm-client:g512xl"  # Specify the Docker image here

