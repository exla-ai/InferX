#!/usr/bin/env python3
"""
Simple script for using the RoboPoint model in the EXLA SDK.
This script demonstrates how to use the RoboPoint model to predict keypoints
for different types of instructions without using argparse.
"""

import os
import sys
from pathlib import Path
import time

# Add the parent directory to the path to import exla
sys.path.append(str(Path(__file__).parent.parent.parent))

from exla.models.robotpoint import robotpoint


def main():
    # Define paths
    script_dir = Path(__file__).parent.absolute()
    base_dir = script_dir.parent.parent  # exla-sdk directory
    image_path = str(base_dir / "sample_images" / "test_table.jpg")
    output_dir = str(script_dir / "output")
    
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return 1
    
    print(f"Using image: {image_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")
    
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
        output_path = os.path.join(output_dir, f"example_{i+1}.png")
        
        print(f"\nRunning example {i+1}: {instruction}")
        start_time = time.time()
        
        result = model.inference(
            image_path=image_path,
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