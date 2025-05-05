## Deployment

In this directory, follow the following steps:

```bash
# Initialize Terraform (downloads providers/modules)
terraform init

# Preview changes (dry-run)
terraform plan

# Apply changes (creates/modifies infrastructure)
terraform apply -auto-approve
```

These commands will create and EC2 instance in AWS.