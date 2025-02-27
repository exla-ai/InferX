#!/usr/bin/env python3
"""
Example script for using the RoboPoint model in the EXLA SDK.
This script demonstrates how to use the RoboPoint model to predict keypoints
for different types of instructions.
"""

import os
import sys
from pathlib import Path
import argparse
import time

# Add the parent directory to the path to import exla
sys.path.append(str(Path(__file__).parent.parent.parent))

from exla.models.robotpoint import robotpoint


def parse_args():
    parser = argparse.ArgumentParser(description="RoboPoint Example")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save output visualizations")
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Check if the image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return 1
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the RoboPoint model
    print("Initializing RoboPoint model...")
    model = robotpoint()
    
    # Define a list of example instructions
    instructions = [
        "Identify points where I can place a cup on the table.",
        "Show me where I could safely grasp this object.",
        "Indicate points where a robot could navigate to in this space.",
        "Mark the corners of the object in the image.",
        "Highlight points along the edge of the surface."
    ]
    
    # Run inference for each instruction
    for i, instruction in enumerate(instructions):
        output_path = os.path.join(args.output_dir, f"example_{i+1}.png")
        
        print(f"\nRunning example {i+1}: {instruction}")
        start_time = time.time()
        
        result = model.inference(
            image_path=args.image,
            text_instruction=instruction,
            output=output_path
        )
        
        elapsed_time = time.time() - start_time
        
        # Print results
        if result["status"] == "success":
            print(f"  Inference successful! ({elapsed_time:.2f} seconds)")
            print(f"  Found {len(result['keypoints'])} keypoints:")
            for j, (x, y) in enumerate(result["keypoints"][:5]):  # Show first 5 keypoints
                print(f"    Keypoint {j+1}: ({x:.4f}, {y:.4f})")
            
            if len(result["keypoints"]) > 5:
                print(f"    ... and {len(result['keypoints']) - 5} more")
                
            if result.get("visualization_path"):
                print(f"  Visualization saved to: {result['visualization_path']}")
        else:
            print(f"  Inference failed: {result.get('error', 'Unknown error')}")
    
    print("\nAll examples completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 