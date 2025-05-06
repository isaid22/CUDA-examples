# variables.tf

variable "region" {
  description = "The AWS region to create resources in"
  type        = string
  default     = "us-east-1"  # You can set a default region or leave it empty if you want to always specify it
}

variable "key_name" {
  description = "The name of the EC2 key pair"
  type        = string
}

variable "subnet_id" {
  description = "The subnet ID in which to launch the instance"
  type        = string
}

variable "security_group_id" {
  description = "The security group ID to associate with the instance"
  type        = string
}

variable "ami_id" {
  description = "The AMI ID to use for the EC2 instance"
  type        = string
}

variable "instance_type" {
  description = "The EC2 instance type"
  type        = string
  default     = "g5.8xlarge"  # You can set a default value for instance type if desired
}

variable "volume_size" {
  description = "The size of the root EBS volume in GB"
  type        = number
  default     = 1024  # Default volume size
}

variable "volume_type" {
  description = "The type of the root EBS volume"
  type        = string
  default     = "gp3"  # Default volume type
}

variable "tags_name" {
  description = "The Name tag for the EC2 instance"
  type        = string
  default     = "MyInstanceName"  # Default value for the Name tag
}

variable "docker_image" {
  description = "The Docker image to pull"
  type        = string
  default     = "rickblaine/vllm-client:g512xl"  # You can specify a default image or leave it empty
}