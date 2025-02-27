# RoboPoint Example

This directory contains a simple script for using the RoboPoint model in the EXLA SDK. The script demonstrates how to use the RoboPoint model to predict keypoints for different types of instructions.

## Overview

RoboPoint is a model that can identify keypoints in images based on natural language instructions. This example shows how to use the model for various tasks such as:

- Identifying points where objects can be placed
- Finding safe grasping points
- Marking navigation points for robots
- Identifying corners of objects
- Highlighting edges of surfaces

## Usage

Simply run the script:

```bash
python3 robopoint_simple.py
```

The script will:
1. Load the RoboPoint model
2. Process a test image with 5 different instructions
3. Save visualizations to the `output` directory

## Output

The script creates an `output` directory containing visualizations for each instruction:
- `example_1.png`: Points where a cup can be placed on a table
- `example_2.png`: Safe grasping points for objects
- `example_3.png`: Navigation points for robots
- `example_4.png`: Corners of objects in the image
- `example_5.png`: Points along the edge of surfaces

## Customization

To modify the script for your own use:
1. Change the image path in the script to use your own images
2. Add or modify the instructions to suit your needs
3. Adjust the output directory if needed

## Requirements

The script requires the following packages:
- numpy
- pillow
- torch
- transformers
- huggingface_hub

These are automatically installed when running the script. 