#!/usr/bin/env python3
"""
Script to generate a test image for the RoboPoint model.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("Error: PIL not found. Please install it with: pip install pillow")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a test image for RoboPoint")
    parser.add_argument("--output_dir", type=str, default="sample_images", help="Directory to save the test image")
    parser.add_argument("--width", type=int, default=640, help="Width of the test image")
    parser.add_argument("--height", type=int, default=480, help="Height of the test image")
    return parser.parse_args()


def generate_table_image(width, height, output_path):
    """Generate a simple image of a table with objects."""
    # Create a blank image with a light gray background
    image = Image.new("RGB", (width, height), (240, 240, 240))
    draw = ImageDraw.Draw(image)
    
    # Draw a table (brown rectangle)
    table_top = height * 0.6
    table_left = width * 0.1
    table_right = width * 0.9
    table_bottom = height * 0.8
    draw.rectangle([(table_left, table_top), (table_right, table_bottom)], fill=(150, 100, 50))
    
    # Draw table legs
    leg_width = width * 0.03
    for x in [table_left + leg_width, table_right - leg_width]:
        draw.rectangle([(x - leg_width/2, table_bottom), (x + leg_width/2, height)], fill=(120, 80, 40))
    
    # Draw some objects on the table
    # Cup
    cup_x = width * 0.3
    cup_y = table_top - height * 0.1
    cup_width = width * 0.06
    cup_height = height * 0.1
    draw.rectangle([(cup_x - cup_width/2, cup_y), (cup_x + cup_width/2, cup_y + cup_height)], fill=(200, 200, 255))
    
    # Book
    book_x = width * 0.5
    book_y = table_top - height * 0.03
    book_width = width * 0.15
    book_height = height * 0.03
    draw.rectangle([(book_x - book_width/2, book_y), (book_x + book_width/2, book_y + book_height)], fill=(255, 200, 200))
    
    # Plate
    plate_x = width * 0.7
    plate_y = table_top - height * 0.02
    plate_radius = width * 0.08
    draw.ellipse([(plate_x - plate_radius, plate_y - plate_radius), 
                  (plate_x + plate_radius, plate_y + plate_radius)], fill=(255, 255, 200))
    
    # Add some text
    draw.text((width * 0.1, height * 0.1), "Test Image for RoboPoint", fill=(0, 0, 0))
    draw.text((width * 0.1, height * 0.15), "Table with objects", fill=(0, 0, 0))
    
    # Save the image
    image.save(output_path)
    print(f"Generated test image saved to: {output_path}")
    return True


def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate the test image
    output_path = os.path.join(args.output_dir, "test_table.jpg")
    success = generate_table_image(args.width, args.height, output_path)
    
    if success:
        print(f"\nYou can now run the RoboPoint example with this image:")
        print(f"  python exla-sdk/examples/robotpoint_example.py --image {output_path}")
        print(f"  python exla-sdk/api/test_api.py --image {output_path} --instruction \"Identify points where I can place a cup on the table\"")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 