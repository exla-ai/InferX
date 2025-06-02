#!/bin/bash
# Simple ECR Push Script - Build and push Docker images to AWS ECR
# Usage: ./ecr_push.sh [image_id_or_name] [repository_name] [tag]

set -e

# Default settings
IMAGE_ID_OR_NAME="${1:-}"
REPOSITORY_NAME="${2:-robopoint-gpu}"
TAG_NAME="${3:-latest}"
PUBLIC_ECR_REGISTRY="public.ecr.aws/h1f5g0k2"
PUBLIC_ECR_REPOSITORY="exla"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Simple ECR Push Tool ===${NC}"

# Function to check Docker installation
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
        echo "Visit https://docs.docker.com/get-docker/ for installation instructions."
        exit 1
    fi
}

# Function to check AWS CLI installation
check_aws_cli() {
    if ! command -v aws &> /dev/null; then
        echo -e "${RED}AWS CLI is not installed. Please install AWS CLI first.${NC}"
        echo "Run: pip install awscli"
        exit 1
    fi
}

# Function to list available Docker images
list_images() {
    echo -e "${YELLOW}Available Docker images:${NC}"
    docker images
}

# Function to push Docker image
push_image() {
    echo -e "${YELLOW}Image ID/Name:${NC} $IMAGE_ID_OR_NAME"
    echo -e "${YELLOW}Repository:${NC} $REPOSITORY_NAME"
    echo -e "${YELLOW}Tag:${NC} $TAG_NAME"
    echo -e "${YELLOW}Public ECR:${NC} $PUBLIC_ECR_REGISTRY/$PUBLIC_ECR_REPOSITORY"
    
    PUBLIC_ECR_URI="$PUBLIC_ECR_REGISTRY/$PUBLIC_ECR_REPOSITORY:$REPOSITORY_NAME-$TAG_NAME"
    
    # If no image ID/name provided, list available images and exit
    if [ -z "$IMAGE_ID_OR_NAME" ]; then
        echo -e "${RED}No image ID or name provided${NC}"
        echo -e "Usage: ./ecr_push.sh [image_id_or_name] [repository_name] [tag]"
        echo -e "Example: ./ecr_push.sh 0a8523455647 robopoint-gpu latest"
        echo -e "Example: ./ecr_push.sh viraatdas/robopoint-gpu:latest robopoint-gpu latest"
        echo ""
        list_images
        exit 1
    fi
    
    # Check if the image exists
    if ! docker image inspect "$IMAGE_ID_OR_NAME" &> /dev/null; then
        echo -e "${RED}Image '$IMAGE_ID_OR_NAME' not found${NC}"
        echo ""
        list_images
        exit 1
    fi
    
    # Tag the image for public ECR
    echo -e "${YELLOW}Tagging image for public ECR: $PUBLIC_ECR_URI${NC}"
    docker tag "$IMAGE_ID_OR_NAME" "$PUBLIC_ECR_URI"
    
    # Export AWS credentials for AWS CLI to use
    export AWS_ACCESS_KEY_ID=""
    export AWS_SECRET_ACCESS_KEY=""
    export AWS_DEFAULT_REGION="us-east-1"
    
    # Login to public ECR
    echo -e "${YELLOW}Logging in to public ECR...${NC}"
    aws ecr-public get-login-password --region "$AWS_DEFAULT_REGION" | docker login --username AWS --password-stdin public.ecr.aws
    
    # Push the image to public ECR
    echo -e "${YELLOW}Pushing image to public ECR...${NC}"
    if docker push "$PUBLIC_ECR_URI"; then
        echo -e "${GREEN}Successfully pushed image to public ECR${NC}"
        
        # Print usage information
        echo -e "\n${BLUE}Your Docker image is now available at:${NC}"
        echo -e "  - Public URI: $PUBLIC_ECR_URI"
        
        echo -e "\n${BLUE}To use in your code:${NC}"
        echo -e "from inferx.models.robopoint import robopoint"
        echo -e "model = robopoint(docker_image=\"$PUBLIC_ECR_URI\")"
    else
        echo -e "${RED}Failed to push image to public ECR${NC}"
        exit 1
    fi
    
    # Clear credentials from environment
    unset AWS_ACCESS_KEY_ID
    unset AWS_SECRET_ACCESS_KEY
    unset AWS_DEFAULT_REGION
}

# Main execution
check_docker
check_aws_cli

# If no arguments provided, list images
if [ -z "$IMAGE_ID_OR_NAME" ]; then
    echo -e "${YELLOW}No image specified. Available images:${NC}"
    list_images
    echo -e "\n${YELLOW}Usage:${NC}"
    echo -e "./ecr_push.sh [image_id_or_name] [repository_name] [tag]"
    echo -e "Example: ./ecr_push.sh 0a8523455647 robopoint-gpu latest"
    exit 0
else
    push_image
fi

echo -e "${GREEN}Operation completed successfully${NC}"

# Security notice
echo -e "\n${YELLOW}SECURITY NOTICE:${NC}"
echo -e "Your AWS credentials were only used for this session and were not stored permanently."
