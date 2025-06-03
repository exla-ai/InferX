# InferX

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


InferX is model wrapper tool we've been using internally to easily test and benchmark ML models across various hardware configurations. It automatically detects your hardware and prepares and executes model inference based on that device. 

This was mainly built to test models on A100/H100s and also Jetsons. Though this can be extended to support any device.

📚 **[View Full Documentation](https://docs.exla.ai/)**

## Key notes:
1. There are different implementations of models.
2. A keynote with InferX is we make sure that only libraries that you need to install per library 

## 🚀 Quick Start

1. **Add your user to the Docker group**
   ```bash
   sudo usermod -aG docker $USER
   ```
   **⚠️ Restart your terminal** for this to take effect.

2. **Install UV**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Clone the examples repository**:
   ```bash
   git clone https://github.com/exla-ai/inferx-examples.git
   cd inferx-examples
   ```

4. **Install InferX**:
   ```bash
   uv pip install git+https://github.com/exla-ai/InferX.git
   ```

5. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate
   ```

6. **Run your first example**:
   ```bash
   python clip/example_clip.py
   ```

🎉 **That's it!** InferX will automatically detect your hardware and run an optimized instance of the model. 

## Examples

Every supported model has a corresponding example here: https://github.com/exla-ai/inferx-examples

## 🎯 Model & Platform Compatibility Matrix

| Model | Jetson AGX Orin | Jetson Orin Nano | NVIDIA H100 | NVIDIA A100 | CPU (x86-64) | CPU (ARM64) |
|-------|----------------|------------------|--------------|-------------|---------------|-------------|
| **CLIP** | ✅ Tested | ✅ Tested | ✅ Tested | ✅ Tested | ✅ Tested | 🟡 Compatible |
| **SAM2** | ✅ Tested | ✅ Tested | 🟡 Compatible | 🟡 Compatible* | 🟡 Compatible | 🟡 Compatible |
| **RoboPoint** | ✅ Tested | ✅ Tested | ✅ Tested | ✅ Tested | ✅ Tested | 🟡 Compatible |
| **Whisper** | ✅ Tested | ✅ Tested | 🟡 Compatible | 🟡 Compatible | ✅ Tested | 🟡 Compatible |
| **DeepSeek R1** | 🟡 Compatible | 🟡 Compatible | ✅ Tested | ✅ Tested | ✅ Tested | 🟡 Compatible |
| **ResNet34** | ✅ Tested | ✅ Tested | 🟡 Compatible | 🟡 Compatible | ✅ Tested | 🟡 Compatible |
| **MobileNet** | ✅ Tested | ✅ Tested | 🟡 Compatible | 🟡 Compatible| ✅ Tested | 🟡 Compatible |
| **InternVL2.5** | 🟡 Compatible | 🟡 Compatible | 🟡 Compatible | ✅ Tested | 🟡 Compatible| 🟡 Compatible |

### **Legend:**
- ✅ **Tested**: Verified working with full optimizations
- 🟡 **Compatible***: Should work based on hardware specs, not extensively tested  
- ❌ **Not Supported**: Hardware limitations

## How It Works

Inferx uses "extreme lazy loading" to only download packages and dependencies you actually need:

1. **Hardware Detection**: The library automatically detects your hardware and selects the appropriate implementation.
2. **On-demand Dependencies**: Dependencies are only installed when needed, reducing installation footprint.
3. **Automatic Setup**: Environment variables and configurations are set up automatically.
4. **Graceful Fallbacks**: If GPU acceleration isn't available, the library *should* (might not be the case for all models currently) falls back to CPU.

## 🏗️ Project Structure

```
inferx/
├── models/                    # Model implementations organized by model type
│   ├── clip/                  # CLIP multimodal model (example)
│   │   ├── __init__.py       # Main CLIP interface
│   │   ├── README.md         # CLIP-specific documentation
│   │   └── _implementations/ # Hardware-specific implementations
│   │       ├── __init__.py   # Implementation selector logic
│   │       ├── clip_gpu.py   # NVIDIA GPU implementation (CUDA + TensorRT)
│   │       ├── clip_jetson.py # Jetson-optimized implementation
│   │       └── clip_cpu.py   # CPU fallback implementation
|   |       ...
├── optimize/                  # Model optimization utilities
├── utils/                    # Hardware detection and utilities
│   ├── __init__.py
│   ├── device_detect.py      # Hardware detection logic
│   ├── docker_utils.py       # Docker container management
│   ├── download_utils.py     # Model download and caching
│   └── config.py             # Configuration management
├── docker/                   # Docker configurations and tools
│   ├── gpu/                  # GPU-specific Docker configs
│   ├── jetson/               # Jetson-specific Docker configs
│   └── cpu/                  # CPU-specific Docker configs
```


## 🤝 Contributing

We welcome contributions! 

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Support

Having issues? Reach out to us at [contact@exla.ai](mailto:contact@exla.ai)

---

**Made with ❤️ for the ML community from Exla**
