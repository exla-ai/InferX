# InferX

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**InferX** is a powerful hardware-optimized inference toolkit that makes it incredibly easy to test and deploy machine learning models across different devices - from edge devices like NVIDIA Jetson to high-performance GPUs and CPUs.

## 🚀 What is InferX?

InferX is the tool we've been using internally to seamlessly test and benchmark ML models across various hardware configurations. It automatically detects your hardware and loads the most optimized implementation for your specific device, making model inference fast and effortless.

### ✨ Key Features

- **🔄 Hardware Auto-Detection**: Automatically detects and optimizes for Jetson, GPU, and CPU environments
- **⚡ Optimized Implementations**: Hardware-specific optimizations for maximum performance
- **🐳 Docker Integration**: Seamless containerized deployment for consistent environments
- **🎯 Model Zoo**: Pre-optimized implementations of popular models (CLIP, SAM2, RoboPoint, DeepSeek, and more)
- **📱 Cross-Platform**: Works on Linux, supports ARM64 and x86_64 architectures
- **🔧 Easy Integration**: Simple Python API that just works

## 🎯 Supported Models

InferX comes with optimized implementations of:

- **CLIP**: Multimodal image-text matching
- **SAM2**: Advanced image segmentation  
- **RoboPoint**: Keypoint affordance prediction for robotics
- **DeepSeek R1**: Large language model capabilities
- **Whisper**: Speech recognition and transcription
- **MobileNet**: Efficient image classification
- **ResNet34**: High-accuracy image classification
- **InternVL2.5**: Vision-language understanding

## 🚀 Quick Start

### Installation

For detailed setup instructions and examples, visit our comprehensive documentation:

**👉 [Complete Setup Guide at docs.exla.ai](https://docs.exla.ai/quickstart)**

### Basic Usage

```python
from inferx.models.clip import clip
import json

# Initialize the model (automatically detects your hardware)
model = clip()

# Run inference
results = model.inference(
    image_paths=["path/to/image1.jpg", "path/to/image2.jpg"],
    text_queries=["a photo of a dog", "a photo of a cat", "a photo of a bird"]
)

print(json.dumps(results, indent=2))
```

### Hardware-Optimized Performance

InferX automatically optimizes for your hardware:

- **NVIDIA Jetson**: TensorRT optimizations, memory-efficient implementations
- **GPU**: CUDA acceleration, batch processing optimizations  
- **CPU**: Optimized for multi-core processing, efficient memory usage

## 🏗️ Project Structure

```
inferx/
├── models/          # Pre-optimized model implementations
│   ├── clip/        # CLIP multimodal model
│   ├── sam2/        # SAM2 segmentation model
│   ├── robopoint/   # RoboPoint keypoint prediction
│   ├── deepseek_r1/ # DeepSeek language model
│   └── ...
├── optimize/        # Model optimization utilities
├── utils/           # Hardware detection and utilities
└── docker/          # Docker configurations and tools
```

## 🐳 Docker Support

InferX includes comprehensive Docker support for containerized deployments:

```bash
# Build optimized containers
cd docker/models/robopoint/gpu
docker build -t inferx/robopoint-gpu:latest .

# Run with GPU support
docker run --gpus all -v /data:/app/data inferx/robopoint-gpu:latest
```

## 🔧 Custom Model Optimization

Easily optimize your own models:

```python
from inferx.optimize import optimize_model

# Optimize your PyTorch model
optimized_model = optimize_model(
    model=your_pytorch_model,
    input_shape=(1, 3, 224, 224),
    device="auto"  # Auto-detects optimal device
)
```

## 🎯 Use Cases

InferX is perfect for:

- **Research & Development**: Quickly test models across different hardware
- **Edge Deployment**: Optimize models for NVIDIA Jetson and ARM devices  
- **Production Inference**: Deploy optimized models in containerized environments
- **Benchmarking**: Compare model performance across different devices
- **Prototyping**: Rapid development with pre-optimized model implementations

## 📊 Performance

InferX delivers significant performance improvements:

- **Up to 3x faster** inference on NVIDIA Jetson with TensorRT optimizations
- **Memory usage reduced by 40%** with efficient implementations
- **Seamless scaling** from edge devices to data center GPUs

## 🤝 Contributing

We welcome contributions! This tool has been essential for our internal ML workflows, and we're excited to share it with the community.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **📚 [Documentation](https://docs.exla.ai/quickstart)** - Complete setup guide and examples
- **🐳 [Docker Hub](https://hub.docker.com/r/inferx)** - Pre-built container images
- **🚀 [Examples Repository](https://github.com/exla-ai/inferx-examples)** - Sample code and tutorials

## 📧 Support

Having issues? Reach out to us at [contact@exla.ai](mailto:contact@exla.ai)

---

**Made with ❤️ for the ML community**
