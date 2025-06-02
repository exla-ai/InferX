# RoboPoint GPU Docker Image

This Docker image provides a containerized environment for running the RoboPoint model with GPU support and 8-bit quantization.

## Features

- CUDA 12.1 support
- 8-bit quantization for efficient inference
- Pre-installed dependencies
- Simple interface for running inference

## Prerequisites

- Docker installed on your system
- NVIDIA Container Toolkit (for GPU support)
- NVIDIA GPU with CUDA support

## Building the Image

To build the Docker image, run the following command from the root of the repository:

```bash
docker build -t exla/robopoint-gpu:latest -f docker/models/robopoint/gpu/Dockerfile .
```

## Pushing the Image

To push the image to a Docker registry, first tag it with your registry information:

```bash
docker tag exla/robopoint-gpu:latest <your-registry>/exla/robopoint-gpu:latest
docker push <your-registry>/exla/robopoint-gpu:latest
```

## Using the Image

### Direct Usage

You can run the container directly:

```bash
docker run --gpus all -v /path/to/input/dir:/app/data/input -v /path/to/output/dir:/app/data/output exla/robopoint-gpu:latest inference /app/data/input/image.jpg "Find keypoints in the image" /app/data/output/result.png
```

### Integration with inferx Package

The RoboPoint model is integrated with the inferx package. After building and pushing the image, you can use it through the Python API:

```python
from examples.models.robopoint.robopoint import robopoint

model = robopoint(docker_image="exla/robopoint-gpu:latest")

result = model.inference(
    image_path="path/to/image.jpg",
    text_instruction="Find keypoints in the image",
    output="path/to/output.png"
)
```

See `examples/models/robopoint/example_docker_robopoint.py` for more examples.

## Parameters

The RoboPoint model accepts the following parameters:

- `image_path`: Path to the input image
- `text_instruction`: Text instruction for the model (optional)
- `output`: Path to save the output visualization (optional)

## Troubleshooting

### GPU Access Issues

If you encounter issues with GPU access, make sure the NVIDIA Container Toolkit is properly installed:

```bash
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Missing Dependencies

The Docker image includes all necessary dependencies. If you encounter any issues, please rebuild the image with the latest Dockerfile.

### Docker Permission Issues

If you encounter permission issues, you may need to run Docker with sudo or add your user to the docker group:

```bash
sudo usermod -aG docker $USER
newgrp docker
``` 
