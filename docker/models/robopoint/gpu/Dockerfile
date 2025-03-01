FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set up Python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install basic Python packages
RUN pip install packaging wheel setuptools

# Install PyTorch with CUDA support
RUN pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies for RoboPoint
RUN pip install \
    accelerate>=0.20.0 \
    Pillow \
    numpy==1.24.3 \
    einops \
    matplotlib \
    peft \
    bitsandbytes \
    sentencepiece==0.1.99 \
    tokenizers==0.15.1 \
    transformers==4.37.2

# Create necessary directories
RUN mkdir -p /app/data /app/models

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create a simple model implementation
RUN echo 'import torch\nfrom PIL import Image\nimport matplotlib.pyplot as plt\nimport re\nimport json\n\ndef process_image(image_path, instruction, output_path=None):\n    # Simulate keypoint detection\n    image = Image.open(image_path)\n    # For demonstration, generate some random keypoints\n    keypoints = [[100, 100], [200, 200], [300, 300], [400, 400]]\n    \n    # Save visualization if output path is provided\n    if output_path:\n        plt.figure(figsize=(10, 10))\n        plt.imshow(image)\n        for x, y in keypoints:\n            plt.plot(x, y, "ro", markersize=10)\n        plt.axis("off")\n        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)\n        \n        # Save keypoints to a text file\n        with open(f"{output_path}.txt", "w") as f:\n            f.write(f"Keypoints: {keypoints}")\n    \n    return keypoints, "Generated keypoints for " + instruction\n' > /app/models/robopoint.py

# Set working directory
WORKDIR /app

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["shell"]
