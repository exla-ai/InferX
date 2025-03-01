# Exla SDK Docker Image Management

This directory contains tools for managing Docker images used by the Exla SDK.

## Simple ECR Push Script

The `ecr_push.sh` script provides a simple way to push Docker images to AWS ECR (Elastic Container Registry). It handles tagging and pushing Docker images to the public ECR repository.

### Public ECR Repository

The script is configured to push images to the following public ECR repository:
```
public.ecr.aws/h1f5g0k2/exla
```

This allows anyone to pull the images without AWS credentials using:
```bash
docker pull public.ecr.aws/h1f5g0k2/exla:robopoint-gpu-latest
```

### Prerequisites

- AWS CLI installed
- Docker installed and running
- AWS credentials configured (via `aws configure` or environment variables)

### Quick Start

1. Make the script executable:
   ```bash
   chmod +x docker/ecr_push.sh
   ```

2. Configure AWS credentials (if not already done):
   ```bash
   aws configure
   ```
   Or set environment variables:
   ```bash
   export AWS_ACCESS_KEY_ID=your-access-key
   export AWS_SECRET_ACCESS_KEY=your-secret-key
   export AWS_DEFAULT_REGION=us-east-1
   ```

3. List available Docker images:
   ```bash
   ./docker/ecr_push.sh
   ```

4. Push an image by ID or name:
   ```bash
   ./docker/ecr_push.sh 0a8523455647 robopoint-gpu latest
   ```
   or
   ```bash
   ./docker/ecr_push.sh viraatdas/robopoint-gpu:latest robopoint-gpu latest
   ```

### Command Format

The script uses a simple command format:
```
./ecr_push.sh [image_id_or_name] [repository_name] [tag]
```

Where:
- `image_id_or_name`: The ID or name of the local Docker image to push (required)
- `repository_name`: The name to use in ECR (defaults to "robopoint-gpu")
- `tag`: The tag to use (defaults to "latest")

### Examples

#### Push by Image ID

If you have an image with ID `0a8523455647`:
```bash
./docker/ecr_push.sh 0a8523455647 robopoint-gpu latest
```

#### Push by Image Name

If you have an image named `viraatdas/robopoint-gpu:latest`:
```bash
./docker/ecr_push.sh viraatdas/robopoint-gpu:latest robopoint-gpu latest
```

#### List Available Images

To see what images are available to push:
```bash
./docker/ecr_push.sh
```

### Tag Conventions

Tags in the ECR repository follow this format:
```
repository-name-tag
```

For example:
- `robopoint-gpu-latest` - The latest version of the robopoint-gpu model
- `robopoint-gpu-v1.0.0` - Version 1.0.0 of the robopoint-gpu model

The `latest` tag is automatically updated when you push without specifying a tag. For versioned releases, use semantic versioning tags like `v1.0.0`, `v1.1.0`, etc.

### The Script Process

The script will:
1. Check if the specified image exists locally
2. Tag it for the public ECR repository
3. Use your configured AWS credentials
4. Log in to ECR and push the image
5. Output the public URI for the pushed image

### Using the Pushed Image

After pushing an image, the script will output the public URI for the image.

#### Pull the image using Docker

```bash
docker pull public.ecr.aws/h1f5g0k2/exla:repository-name-tag
```

#### Use in Exla SDK code

```python
from exla.models.robopoint import robopoint
model = robopoint(docker_image="public.ecr.aws/h1f5g0k2/exla:repository-name-tag")
```

### Security Best Practices

- Configure AWS credentials using `aws configure` or environment variables
- Use IAM roles with minimal required permissions
- For better security in production environments, consider using IAM roles
- Never share scripts with hardcoded credentials
- Regularly rotate your AWS access keys

### Troubleshooting

- **AWS CLI not installed**: Install AWS CLI with `pip install awscli`
- **Docker not running**: Start Docker with `systemctl start docker` or the appropriate command for your OS
- **Permission denied**: Ensure you have the necessary AWS permissions to push to the public ECR repository
- **Image not found**: Make sure you're using the correct image ID or name (run the script without arguments to list available images)
- **AWS credentials not found**: Run `aws configure` to set up your credentials or set the appropriate environment variables

## Workflow for Adding a New Model

1. **Create your Dockerfile** in the appropriate directory:
   ```bash
   mkdir -p docker/models/my-new-model
   # Create your Dockerfile in this directory
   ```

2. **Build and push to ECR**:
   ```bash
   ./docker/ecr_push.sh --repository my-new-model
   ```

3. **Update your model code** to use the public ECR URI:
   ```python
   # In your model implementation
   DEFAULT_DOCKER_REPO = "public.ecr.us-east-1.amazonaws.com/my-new-model"
   ```

## Required AWS Permissions

To use this script, your AWS user needs the following permissions:

1. `ecr-public:GetAuthorizationToken` - For logging in to the public ECR
2. `ecr-public:BatchCheckLayerAvailability` - For checking if layers exist
3. `ecr-public:PutImage` - For pushing images
4. `ecr-public:InitiateLayerUpload` - For uploading layers
5. `ecr-public:UploadLayerPart` - For uploading layer parts
6. `ecr-public:CompleteLayerUpload` - For completing layer uploads

You can create an IAM policy with these permissions and attach it to your user or role. 