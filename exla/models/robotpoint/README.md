# RoboPoint Model for EXLA SDK

RoboPoint is a Vision-Language Model for Spatial Affordance Prediction for Robotics. It predicts image keypoint affordances given language instructions.

## Overview

RoboPoint is a VLM that predicts image keypoint affordances given language instructions. It provides a generic action space that enables language-conditioned task execution in several downstream applications such as robot navigation, manipulation, and augmented reality (AR) assistance.

## Features

- Predict keypoints based on image and text instruction
- Visualize predicted keypoints on images
- Support for different hardware platforms (CPU, GPU, Jetson)
- Automatic hardware detection and optimization

## Usage

### Basic Usage

```python
from exla.models.robotpoint import robotpoint

# Initialize the model
model = robotpoint()

# Run inference
result = model.inference(
    image_path="path/to/image.jpg",
    text_instruction="Identify points where I can place a cup on the table",
    output="output.png"
)

# Access the results
if result["status"] == "success":
    keypoints = result["keypoints"]
    visualization_path = result["visualization_path"]
    print(f"Found {len(keypoints)} keypoints")
else:
    print(f"Error: {result.get('error', 'Unknown error')}")
```

### Command Line Usage

You can also use the provided test script to run inference:

```bash
python exla-sdk/api/test_api.py --image path/to/image.jpg --instruction "Identify points where I can place a cup on the table" --output output.png
```

## Model Variants

RoboPoint comes with different model variants optimized for different hardware:

- **CPU**: Uses the full model on CPU (slower but works everywhere)
- **GPU**: Uses the full model with GPU acceleration and optional quantization
- **Jetson**: Uses a smaller model with LoRA weights optimized for edge devices

The appropriate model is automatically selected based on your hardware.

## References

- [RoboPoint GitHub Repository](https://github.com/wentaoyuan/RoboPoint)
- [RoboPoint Paper](https://arxiv.org/abs/2406.10721) 