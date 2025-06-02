# RoboPoint Model

RoboPoint is a Vision-Language Model for Spatial Affordance Prediction for Robotics. It predicts image keypoint affordances given language instructions.

## Overview

RoboPoint is a VLM that predicts image keypoint affordances given language instructions. It provides a generic action space that enables language-conditioned task execution in several downstream applications such as robot navigation, manipulation, and augmented reality (AR) assistance.

## Installation

The RoboPoint model will automatically install the required dependencies when initialized. However, if you prefer to install them manually, you can use the following commands:

```bash
# Install PyTorch with CUDA support
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118

# Install other required packages
pip install transformers>=4.31.0 accelerate>=0.21.0 bitsandbytes>=0.41.0 pillow>=9.0.0 numpy>=1.24.0 huggingface_hub>=0.16.4
```

Alternatively, you can use the provided installation script:

```bash
./examples/robopoint/install_dependencies.sh
```

## Usage

Here's a simple example of how to use the RoboPoint model:

```python
from inferx.models.robopoint import robopoint

# Initialize the model (will automatically install dependencies if needed)
model = robopoint(
    temperature=0.7,  # Sampling temperature (0.0 to 1.0)
    top_p=0.9,        # Nucleus sampling threshold (0.0 to 1.0)
    max_output_tokens=100,  # Maximum number of tokens to generate
    force_gpu=True    # Force GPU implementation
)

# Run inference
result = model.inference(
    image_path="path/to/image.jpg",
    text_instruction="Find a few spots within the vacant area on the table.",
    output="path/to/output.png"  # Optional: save visualization
)

# Access the results
keypoints = result["keypoints"]  # List of (x, y) tuples
visualization_path = result["visualization_path"]  # Path to saved visualization
resources = result["resources"]  # Resource usage information
```

## Automatic Dependency Installation

When you initialize the RoboPoint model, it will:

1. Check for required dependencies from the `requirements_gpu.txt` file
2. Install PyTorch with CUDA support if needed
3. Install any missing dependencies automatically
4. Fall back to a mock implementation if installation fails

This ensures that the model works out of the box without manual setup.

## Model Details

- **Model Architecture**: Based on the Vicuna-v1.5-13b language model with CLIP vision encoder
- **Quantization**: Supports 8-bit and 4-bit quantization for reduced memory usage
- **Hardware Support**: Optimized implementations for GPU, CPU, and Jetson devices

## Example

See the `examples/robopoint/example_robopoint.py` script for a complete example of how to use the RoboPoint model.

## Citation

If you use RoboPoint in your research, please cite the original paper:

```
@article{yuan2024robopoint,
  title={RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics},
  author={Yuan, Wentao and Duan, Jiafei and Blukis, Valts and Pumacay, Wilbert and Krishna, Ranjay and Murali, Adithyavairavan and Mousavian, Arsalan and Fox, Dieter},
  journal={arXiv preprint arXiv:2406.10721},
  year={2024}
}
```

## License

This implementation is provided under the Apache 2.0 license.

## Troubleshooting CUDA Issues

If you encounter issues with CUDA setup, especially with the bitsandbytes library, you can use our CUDA setup helper script:

```bash
python examples/robopoint/setup_cuda.py
```

This script will:
1. Check if CUDA is available through PyTorch
2. Find CUDA libraries on your system
3. Set up the LD_LIBRARY_PATH environment variable
4. Verify and fix bitsandbytes installation

Common CUDA issues and solutions:

1. **Missing CUDA libraries**: The most common issue is that the CUDA runtime libraries (like `libcudart.so`) are not in your `LD_LIBRARY_PATH`. The setup script will help find these libraries and create a shell script to set the correct environment variables.

2. **Incompatible bitsandbytes version**: The bitsandbytes library needs to match your CUDA version. The setup script will attempt to install the correct version.

3. **Multiple CUDA installations**: If you have multiple CUDA installations, there might be conflicts. The setup script will help identify and resolve these conflicts.

If you still encounter issues after running the setup script, you can try:

```bash
# Source the environment script created by the setup helper
source examples/robopoint/set_cuda_env.sh

# Run the example script
python examples/robopoint/example_robopoint.py
```

For more detailed information about bitsandbytes CUDA setup, see the [bitsandbytes documentation](https://github.com/TimDettmers/bitsandbytes). 