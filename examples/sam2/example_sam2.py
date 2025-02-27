#!/usr/bin/env python3
"""
SAM2 Example - Camera Simulation

This script demonstrates how to use the SAM2 model with a simulated camera input
using a pre-recorded video file. It shows how to process video frames and
visualize the segmentation results in real-time.

Usage:
    python example_sam2.py [--video VIDEO_PATH] [--output OUTPUT_PATH] [--server]
"""

import argparse
import os
import time
from pathlib import Path
import sys

# Check for required packages and install if missing
required_packages = ['numpy', 'cv2', 'matplotlib']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f"Missing required packages: {', '.join(missing_packages)}")
    print("Installing missing packages...")
    
    package_map = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib'
    }
    
    for package in missing_packages:
        pkg_name = package_map.get(package, package)
        print(f"Installing {pkg_name}...")
        try:
            import subprocess
            # Try without --user flag first
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
            print(f"Successfully installed {pkg_name}")
        except Exception as e:
            print(f"Failed to install {pkg_name}: {str(e)}")
            if package == 'cv2':
                print("OpenCV is required for this example. Please install it manually:")
                print("pip install opencv-python")
                sys.exit(1)

# Now import the required packages
import numpy as np
try:
    import cv2
except ImportError:
    print("Failed to import OpenCV (cv2) even after installation attempt.")
    print("Please install it manually: pip install opencv-python")
    sys.exit(1)

try:
    # Import the SAM2 model
    from exla.models.sam2 import sam2
    from exla.models.sam2._implementations.sam2_jetson import SAM2_Jetson
except ImportError as e:
    print(f"Error importing SAM2 model: {str(e)}")
    print("Make sure you're running this script from the correct directory.")
    print("Try running: cd /path/to/exla-sdk && python examples/sam2/example_sam2.py")
    sys.exit(1)

def create_sam2_model(use_server=False):
    """
    Create a SAM2 model instance.
    
    Args:
        use_server: Whether to use the server or local model
        
    Returns:
        SAM2 model instance
    """
    try:
        if use_server:
            # Use the factory function which may connect to the server
            model = sam2()
            print("✓ Successfully initialized SAM2 model")
            
            # Check if we're actually using the server
            if getattr(model, 'use_server', False):
                print("Using SAM2 server mode")
            else:
                print("Server not available, using local model")
                # Check if model files are available
                if not hasattr(model, 'predictor') or model.predictor is None:
                    print("Warning: SAM2 predictor is not available. This may be due to missing model files.")
                    print("Please make sure you have downloaded the SAM2 model files.")
                    print("You can download the SAM2 model from: https://github.com/facebookresearch/segment-anything-2#model-checkpoints")
        else:
            # Directly initialize the Jetson implementation
            model = SAM2_Jetson(model_name="sam2_b")
            # Force local mode
            model.use_server = False
            print("✓ Successfully initialized SAM2 model in local mode")
            
            # Check if the predictor is available
            if not hasattr(model, 'predictor') or model.predictor is None:
                print("Warning: SAM2 predictor is not available. This may be due to missing model files.")
                print("Please make sure you have downloaded the SAM2 model files.")
                print("You can download the SAM2 model from: https://github.com/facebookresearch/segment-anything-2#model-checkpoints")
                print(f"Model files should be placed in: {model.cache_dir if hasattr(model, 'cache_dir') else '~/.cache/exla/sam2'}")
                print(f"Expected model file: {model.checkpoint_name if hasattr(model, 'checkpoint_name') else 'sam2_b.pth'}")
            
        return model
    except Exception as e:
        print(f"❌ Error initializing SAM2 model: {str(e)}")
        print("Make sure you have downloaded the SAM2 model files.")
        print("You can download the SAM2 model from: https://github.com/facebookresearch/segment-anything-2#model-checkpoints")
        return None

def simulate_camera_with_video(video_path, output_path=None, display=True, use_server=False):
    """
    Simulate camera input using a video file and run SAM2 segmentation on each frame.
    
    Args:
        video_path: Path to the input video file
        output_path: Path to save the output video (optional)
        display: Whether to display the output in a window
        use_server: Whether to use the server or local model
    """
    print(f"Simulating camera with video: {video_path}")
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        print(f"Current working directory: {os.getcwd()}")
        print("Please provide a valid video file path.")
        return
    
    # Initialize the SAM2 model
    model = create_sam2_model(use_server)
    if model is None:
        return
    
    # Check if we're using server mode (camera not supported in server mode)
    if getattr(model, 'use_server', False):
        print("Camera simulation not supported in server mode.")
        print("Please run without --server flag to use local model.")
        return
    
    # Check if predictor is available
    if not hasattr(model, 'predictor') or model.predictor is None:
        print("Error: SAM2 predictor is not available. Cannot process video.")
        print("This may be due to missing model files or initialization issues.")
        return
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Setup output video writer if needed
    video_writer = None
    if output_path:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Setup display window if needed
    if display:
        try:
            cv2.namedWindow("SAM2 Segmentation", cv2.WINDOW_NORMAL)
        except Exception as e:
            print(f"Warning: Could not create display window: {str(e)}")
            print("Running in headless mode (no display)")
            display = False
    
    # Define tracking point in the center of the frame
    center_point = {
        "points": [[width//2, height//2]],
        "labels": [1]  # Foreground point
    }
    
    # Process frames
    frame_count = 0
    processing_times = []
    avg_time = 0
    
    print("Processing video frames with SAM2. Press 'q' to quit.")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Start timing
        start_time = time.time()
        
        # Save frame to temporary file (SAM2 expects a file path)
        temp_frame_path = "temp_frame.jpg"
        cv2.imwrite(temp_frame_path, frame)
        
        try:
            # Process frame with SAM2
            # Convert frame to RGB for SAM2
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Set image in predictor
            model.predictor.set_image(frame_rgb)
            
            # Process with center point
            point_coords = np.array(center_point["points"])
            point_labels = np.array(center_point["labels"])
            
            masks, scores, _ = model.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True
            )
            
            # Use the highest scoring mask
            best_mask_idx = np.argmax(scores)
            mask = masks[best_mask_idx]
            
            # Create visualization
            mask_overlay = np.zeros_like(frame)
            mask_overlay[mask] = [0, 114, 255]  # BGR format for OpenCV
            
            # Draw the center point
            for coord, label in zip(point_coords, point_labels):
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                cv2.drawMarker(frame, (int(coord[0]), int(coord[1])), color, 
                              markerType=cv2.MARKER_STAR, markerSize=20, thickness=2)
            
            # Blend with original frame
            result = cv2.addWeighted(frame, 0.7, mask_overlay, 0.3, 0)
            
            # Add processing time text
            process_time = time.time() - start_time
            processing_times.append(process_time)
            avg_time = sum(processing_times) / len(processing_times)
            cv2.putText(result, f"SAM2 | Frame: {frame_count} | Time: {process_time:.3f}s | Avg: {avg_time:.3f}s", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Write to output video if needed
            if video_writer:
                video_writer.write(result)
            
            # Display if needed
            if display:
                cv2.imshow("SAM2 Segmentation", result)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        except Exception as e:
            print(f"❌ Error processing frame {frame_count} with SAM2: {str(e)}")
            # Display error on frame
            cv2.putText(frame, f"SAM2 Error: {str(e)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if display:
                cv2.imshow("SAM2 Segmentation", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            
            if video_writer:
                video_writer.write(frame)
        
        # Clean up temporary file
        if os.path.exists(temp_frame_path):
            os.remove(temp_frame_path)
        
        frame_count += 1
        
        # Print progress every 10 frames
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames with SAM2. Average time: {avg_time:.3f}s per frame")
    
    # Clean up
    cap.release()
    if video_writer:
        video_writer.release()
    if display:
        cv2.destroyAllWindows()
    
    print(f"SAM2 processing complete: {frame_count} frames processed")
    if len(processing_times) > 0:
        print(f"Average SAM2 processing time: {avg_time:.3f}s per frame")
    
    # Remove temporary file if it exists
    if os.path.exists(temp_frame_path):
        os.remove(temp_frame_path)

def process_image(image_path, output_path=None, use_server=False):
    """
    Process a single image with SAM2.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the output image (optional)
        use_server: Whether to use the server or local model
    """
    print(f"Processing image with SAM2: {image_path}")
    
    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        print(f"Current working directory: {os.getcwd()}")
        print("Please provide a valid image file path.")
        return
    
    # Initialize the SAM2 model
    model = create_sam2_model(use_server)
    if model is None:
        return
    
    # Create output directory if needed
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    # Check if we're using server mode or if the predictor is available
    if getattr(model, 'use_server', False) or (hasattr(model, 'predictor') and model.predictor is not None):
        # Run inference
        try:
            result = model.inference(
                input=image_path,
                output=output_path
            )
            
            print(f"SAM2 inference result: {result['status']}")
            print(f"Found {len(result.get('masks', []))} masks")
            
            if output_path:
                print(f"SAM2 output saved to: {output_path}")
        except Exception as e:
            print(f"❌ Error during inference: {str(e)}")
    else:
        print("Error: SAM2 predictor is not available and server mode is not enabled.")
        print("This is likely because the SAM2 model files are not downloaded.")
        print("Please download the SAM2 model files from: https://github.com/facebookresearch/segment-anything-2#model-checkpoints")
        print(f"And place them in: {model.cache_dir if hasattr(model, 'cache_dir') else '~/.cache/exla/sam2'}")
        print("Or use --server flag to use the server mode instead.")

def main():
    parser = argparse.ArgumentParser(description="SAM2 Example - Camera Simulation")
    parser.add_argument("--video", type=str, default="data/f1_trimmed.mp4",
                        help="Path to input video file (default: data/f1_trimmed.mp4)")
    parser.add_argument("--image", type=str, default="data/truck.jpg",
                        help="Path to input image file (default: data/truck.jpg)")
    parser.add_argument("--output", type=str, default="data/output.mp4",
                        help="Path to save output video (default: data/output.mp4)")
    parser.add_argument("--mode", type=str, default="video", choices=["video", "image"],
                        help="Processing mode: video or image (default: video)")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable display window")
    parser.add_argument("--server", action="store_true",
                        help="Use server mode instead of local model")
    
    args = parser.parse_args()
    
    print("SAM2 Example - Camera Simulation")
    print("================================")
    print(f"Current working directory: {os.getcwd()}")
    
    if args.mode == "video":
        simulate_camera_with_video(
            video_path=args.video,
            output_path=args.output,
            display=not args.no_display,
            use_server=args.server
        )
    else:
        output_path = args.output.replace(".mp4", ".jpg") if args.output else "data/output.jpg"
        process_image(
            image_path=args.image,
            output_path=output_path,
            use_server=args.server
        )

if __name__ == "__main__":
    main() 