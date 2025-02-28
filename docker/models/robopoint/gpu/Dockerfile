FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/root/.local/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    git \
    wget \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install basic dependencies first
RUN pip install packaging wheel setuptools

# Install PyTorch with CUDA support
RUN pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu121

# Install RoboPoint dependencies
RUN pip install \
    transformers==4.31.0 \
    bitsandbytes>=0.39.0 \
    accelerate>=0.20.0 \
    Pillow \
    numpy==1.24.3 \
    einops \
    sentencepiece

# Create directories for model and data
RUN mkdir -p /app/data
RUN mkdir -p /app/exla/utils
RUN mkdir -p /app/exla/models/robopoint/_implementations

# Set working directory
WORKDIR /app

# Create a minimal exla module structure
RUN echo 'def detect_device():\n    import torch\n    if torch.cuda.is_available():\n        return "cuda"\n    else:\n        return "cpu"' > /app/exla/utils/device_detect.py
RUN touch /app/exla/__init__.py
RUN touch /app/exla/utils/__init__.py
RUN touch /app/exla/models/__init__.py
RUN touch /app/exla/models/robopoint/__init__.py
RUN touch /app/exla/models/robopoint/_implementations/__init__.py

# Create a simple mock implementation for testing
RUN echo 'class Robopoint_Base:\n    def __init__(self):\n        pass\n\n    def inference(self, image_path, text_instruction=None, output=None):\n        """Run inference with the RoboPoint model to predict keypoint affordances."""\n        print(f"Running inference on {self.__class__.__name__}")\n        from PIL import Image, ImageDraw\n        import random\n        # Load the image\n        img = Image.open(image_path)\n        draw = ImageDraw.Draw(img)\n        # Generate random keypoints\n        width, height = img.size\n        keypoints = []\n        for _ in range(5):\n            x = random.randint(0, width-1)\n            y = random.randint(0, height-1)\n            keypoints.append((x, y))\n            # Draw a circle at each keypoint\n            draw.ellipse((x-5, y-5, x+5, y+5), fill="red", outline="red")\n        # Save the output\n        if output:\n            img.save(output)\n            print(f"Saved output to {output}")\n        return {"status": "success", "keypoints": keypoints}' > /app/exla/models/robopoint/_implementations/_base.py

RUN echo 'from ._base import Robopoint_Base\nfrom exla.utils.device_detect import detect_device\nimport torch\nimport os\nimport sys\n\nclass RoboPointGPU(Robopoint_Base):\n    def __init__(self, model_name="liuhaotian/llava-v1.5-7b", device="cuda" if torch.cuda.is_available() else "cpu", use_8bit=True, **kwargs):\n        super().__init__()\n        self.model_name = model_name\n        self.device = device\n        self.use_8bit = use_8bit\n        print(f"Initializing RoboPoint GPU implementation with model: {model_name}")\n        print(f"Device: {device}, Use 8-bit: {use_8bit}")\n        print(f"Using Python {sys.version} environment at: {sys.prefix}")\n        \n        # Skip actual model loading in this mock implementation\n        # Just use the base implementation for testing\n\n    def inference(self, image_path, text_instruction=None, output=None):\n        print(f"Running inference on RoboPointGPU")\n        print(f"Image: {image_path}")\n        print(f"Instruction: {text_instruction}")\n        print(f"Output: {output}")\n        return super().inference(image_path, text_instruction, output)' > /app/exla/models/robopoint/_implementations/robopoint_gpu.py

RUN echo 'from ._implementations.robopoint_gpu import RoboPointGPU\n\ndef robopoint(**kwargs):\n    """Factory function to create a RoboPoint model instance."""\n    import torch\n    if torch.cuda.is_available():\n        print("CUDA is available, using GPU implementation")\n        return RoboPointGPU(**kwargs)\n    else:\n        print("CUDA is not available, using CPU implementation")\n        return RoboPointGPU(device="cpu", **kwargs)\n\n# Alias for backward compatibility\nRoboPoint = robopoint\n\n__all__ = ["robopoint", "RoboPoint"]' > /app/exla/models/robopoint/__init__.py

# Create example directory and files
RUN mkdir -p /app/examples/robopoint/data

# Copy example images (assuming they exist in the build context)
COPY examples/robopoint/data/sink.jpg /app/examples/robopoint/data/sink.jpg
COPY examples/robopoint/data/stair.jpg /app/examples/robopoint/data/stair.jpg

# Create example script
RUN echo 'import os\nimport sys\n\n# Add the parent directory to the Python path\nsys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))\n\nfrom exla.models.robopoint import robopoint\n\n# Get the current directory\ncurrent_dir = os.path.dirname(os.path.abspath(__file__))\n\nmodel = robopoint()\n\nmodel.inference(\n    image_path=os.path.join(current_dir, "data/sink.jpg"),\n    text_instruction="Find a few spots within the vacant area on the rightmost white plate.",\n    output=os.path.join(current_dir, "data/sink_output.png")\n)\n\nmodel.inference(\n    image_path=os.path.join(current_dir, "data/stair.jpg"),\n    text_instruction="Identify several places in the unoccupied space on the stair in the middle.",\n    output=os.path.join(current_dir, "data/stair_output.png")\n)' > /app/examples/robopoint/example_robopoint.py

# Create entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$1" = "inference" ]; then\n\
    shift\n\
    python -c "import sys; sys.path.insert(0, \"/app\"); from exla.models.robopoint import robopoint; model = robopoint(use_8bit=True); model.inference(image_path=\"$1\", text_instruction=\"$2\", output=\"$3\")" \n\
elif [ "$1" = "example" ]; then\n\
    cd /app/examples/robopoint && python example_robopoint.py\n\
else\n\
    exec "$@"\n\
fi' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["inference", "/app/data/input.jpg", "Find keypoints in the image", "/app/data/output.png"]
