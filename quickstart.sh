#!/bin/bash

# Install the package in development mode
uv pip install -e .

# Run the example to test device detection
python examples/mobilenet/example_mobilenet.py