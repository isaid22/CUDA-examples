provider "aws" {
  region = var.region
}

# Reference an existing security group (instead of trying to create it)
data "aws_security_group" "my_sg" {
  id = var.security_group_id  # Your existing SG ID
}

resource "aws_instance" "g5_8xlarge" {
  ami           = var.ami_id  # Removed duplicate "ami-" prefix
  instance_type = var.instance_type  # Fixed from g5.xlarge to g5.8xlarge
  key_name      = var.key_name   # Key pair name

  # Network config
  subnet_id                   = var.subnet_id  # Subnet implies VPC
  vpc_security_group_ids      = [data.aws_security_group.my_sg.id]  # Use data source
  associate_public_ip_address = true

  # Root volume configuration (correct block type)
  root_block_device {
    volume_size           = var.volume_size  # 1024GB
    volume_type           = var.volume_type
    delete_on_termination = true
  }

    # User data to install Docker and run commands
  user_data = <<-EOF
              #!/bin/bash
              # Update and install Docker
              apt-get update -y
              apt-get install -y docker.io

              # Start Docker service
              systemctl start docker
              systemctl enable docker

              # Pull Docker image
              docker pull ${var.docker_image}
            EOF

  tags = {
    Name = var.tags_name
  }
}
