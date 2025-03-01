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

# Install PyTorch with CUDA support - using a version compatible with CUDA 12.1
RUN pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# Install RoboPoint dependencies
RUN pip install \
    accelerate>=0.20.0 \
    Pillow \
    numpy==1.24.3 \
    einops \
    sentencepiece \
    safetensors \
    protobuf \
    regex \
    bitsandbytes>=0.41.0 \
    peft>=0.4.0

# Install transformers from GitHub to get LLaVA support
RUN pip install git+https://github.com/huggingface/transformers.git

# Create directories for model and data
RUN mkdir -p /app/data
WORKDIR /app

# Copy the entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["shell"]
