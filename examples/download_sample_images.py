#!/usr/bin/env python3
"""
Script to download sample images for testing the RoboPoint model.
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Download sample images for RoboPoint")
    parser.add_argument("--output_dir", type=str, default="sample_images", help="Directory to save sample images")
    return parser.parse_args()


def download_image(url, output_path):
    """Download an image from a URL to the specified output path."""
    try:
        print(f"Downloading {url} to {output_path}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded successfully!")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # List of sample images to download
    sample_images = [
        {
            "name": "table_with_objects.jpg",
            "url": "https://raw.githubusercontent.com/microsoft/TaskMatrix/main/assets/demo/demo_1.jpg",
            "description": "Table with various objects for placement tasks"
        },
        {
            "name": "kitchen_counter.jpg",
            "url": "https://raw.githubusercontent.com/microsoft/TaskMatrix/main/assets/demo/demo_2.jpg",
            "description": "Kitchen counter for object placement and manipulation"
        },
        {
            "name": "room_navigation.jpg",
            "url": "https://raw.githubusercontent.com/microsoft/TaskMatrix/main/assets/demo/demo_3.jpg",
            "description": "Room for navigation tasks"
        }
    ]
    
    # Download each image
    successful_downloads = 0
    for image in sample_images:
        output_path = os.path.join(args.output_dir, image["name"])
        if download_image(image["url"], output_path):
            successful_downloads += 1
    
    # Print summary
    print(f"\nDownloaded {successful_downloads} out of {len(sample_images)} images to {args.output_dir}/")
    
    if successful_downloads > 0:
        print("\nYou can now run the RoboPoint example with one of these images:")
        for image in sample_images[:successful_downloads]:
            print(f"  python exla-sdk/examples/robotpoint_example.py --image {args.output_dir}/{image['name']}")
            print(f"    Description: {image['description']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 