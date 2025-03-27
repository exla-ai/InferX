#!/bin/bash

# Exit on error
set -e

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv package installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv venv --python 3.10

source .venv/bin/activate

# Update pip and install packages
echo "Installing required packages..."


echo "Setup complete! Activate the virtual environment if not already activated with: source .venv/bin/activate"
uv pip install -r pyproject.toml
uv pip install -e .
