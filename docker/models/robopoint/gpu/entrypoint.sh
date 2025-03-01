#!/bin/bash
set -e

echo "Starting entrypoint script with arguments: $@"

case "$1" in
    inference)
        echo "Running inference with arguments: $2 $3 $4"
        # Run inference with the provided arguments
        python -c "
import sys
import os
from models.robopoint import process_image

# Parse arguments
image_path = '$2'
instruction = '$3'
output_path = '$4'

print(f'Processing image: {image_path}')
print(f'Instruction: {instruction}')
print(f'Output path: {output_path}')

try:
    # Process the image using our custom implementation
    keypoints, response = process_image(image_path, instruction, output_path)
    
    print(f'Keypoints: {keypoints}')
    print(f'Raw response: {response}')
    
    if output_path:
        print(f'Visualization saved to {output_path}')
        print(f'Keypoints saved to {output_path}.txt')
except Exception as e:
    print(f'Error during inference: {e}')
    sys.exit(1)
"
        ;;
    shell)
        echo "Starting a shell"
        # Start a shell
        exec /bin/bash
        ;;
    *)
        echo "Unknown command: $1"
        echo "Available commands: inference, shell"
        exit 1
        ;;
esac 