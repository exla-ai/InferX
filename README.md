# InferX

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


InferX is model wrapper tool we've been using internally to seamlessly test and benchmark ML models across various hardware configurations. It automatically detects your hardware and loads the most optimized implementation for certain devices, making model inference fast and effortless.

This was mainly built to run devices across A100/H100s and also Jetsons. 


## 🎯 Supported Models

InferX comes with these models

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


## 🤝 Contributing

We welcome contributions! Please add more models

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Support

Having issues? Reach out to us at [contact@exla.ai](mailto:contact@exla.ai)

---

**Made with ❤️ for the ML community from Exla**
