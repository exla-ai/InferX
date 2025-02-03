#!/bin/bash

# Exit on error
set -e

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv package installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Update pip and install packages
echo "Installing required packages..."
uv pip install --upgrade pip
uv pip install -r requirements.txt

# Verify CUDA is available
echo "Verifying CUDA installation..."
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "Setup complete! Activate the virtual environment with: source .venv/bin/activate"

