from ._base import SAM2_Base
import os
import subprocess
import time
import sys
import json
from pathlib import Path
import tempfile
import random

class SAM2_GPU(SAM2_Base):
    def __init__(self, port=8080):
        """
        Initializes a SAM2 model on GPU using Docker.
        
        Args:
            port (int): Port to run the SAM2 server on
        """
        super().__init__()
        self._port = port
        self._container_name = f"exla-sam2-server-{random.randint(1000, 9999)}"
        self._client = None
        
        # Start the server
        self._start_server()
        
        # Initialize the client
        self._init_client()
        
        # Install dependencies
        self._install_dependencies()
        
    def _install_dependencies(self):
        """Install required dependencies for SAM2 client"""
        try:
            # Check if dependencies are already installed
            import requests
            print("SAM2 client dependencies already installed")
            return
        except ImportError:
            pass
            
        print("Installing SAM2 client dependencies...")
        subprocess.run([
            "uv", "pip", "install", "requests", "pillow", "numpy", "opencv-python"
        ], check=True)
        print("Successfully installed SAM2 client dependencies")
        
    def _start_server(self):
        """Start the SAM2 server using Docker."""
        try:
            print(f"Starting SAM2 server on port {self._port}...")
            
            # Check if Docker is available
            subprocess.run(["docker", "--version"], check=True, stdout=subprocess.PIPE)
            
            # Stop any existing container with the same name
            subprocess.run(["docker", "rm", "-f", self._container_name], 
                          stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            
            # Start the container
            subprocess.run([
                "docker", "run", "-d",
                "--name", self._container_name,
                "-p", f"{self._port}:8080",
                "--gpus", "all",
                "samhub/sam2:latest"  # This is a placeholder - use the actual SAM2 Docker image
            ], check=True)
            
            # Wait for the server to start
            time.sleep(5)
            print("SAM2 server started successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to start SAM2 server: {str(e)}")
            
    def _init_client(self):
        """Initialize the SAM2 client."""
        try:
            import requests
            
            # Test connection to server
            response = requests.get(f"http://localhost:{self._port}/health")
            if response.status_code == 200:
                print("Connected to SAM2 server successfully")
            else:
                raise RuntimeError(f"Server returned status code: {response.status_code}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to connect to SAM2 server: {str(e)}")
    
    def _process_image(self, image_path, output_path=None, prompt=None):
        """
        Process a single image with SAM2 server.
        
        Args:
            image_path (str): Path to input image
            output_path (str, optional): Path to save the output
            prompt (dict, optional): Prompt for the model
            
        Returns:
            dict: Results from the segmentation
        """
        import requests
        from PIL import Image
        import numpy as np
        import json
        import base64
        import io
        import cv2
        
        # Load image
        print(f"Loading image: {image_path}")
        image = Image.open(image_path)
        
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Prepare request data
        request_data = {
            "image": img_str,
        }
        
        # Add prompt if provided
        if prompt:
            request_data["prompt"] = prompt
            
        # Send request to server
        response = requests.post(
            f"http://localhost:{self._port}/predict",
            json=request_data
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Server returned status code: {response.status_code}")
            
        # Parse response
        result = response.json()
        
        # Save output if specified
        if output_path and "masks" in result:
            print(f"Saving output to: {output_path}")
            
            # Convert masks to numpy arrays
            masks = [np.array(mask) for mask in result["masks"]]
            
            # Load original image for blending
            original_image = cv2.imread(image_path)
            
            # Create a visualization of the masks
            mask_image = np.zeros_like(original_image)
            for i, mask in enumerate(masks):
                color = np.random.randint(0, 255, size=3)
                mask_image[mask] = color
            
            # Blend with original image
            result_image = cv2.addWeighted(original_image, 0.7, mask_image, 0.3, 0)
            
            # Save the result
            cv2.imwrite(output_path, result_image)
        
        return result
            
    def inference(self, input, output=None, prompt=None):
        """
        Run inference with the SAM2 model.
        
        Args:
            input (str): Path to input image or video
            output (str, optional): Path to save the output
            prompt (dict, optional): Prompt for the model (points, boxes, etc.)
            
        Returns:
            dict: Results from the segmentation
        """
        try:
            print("\n🚀 Starting Exla SAM2 Server Inference Pipeline\n")
            
            # Determine if input is image or video based on extension
            input_path = Path(input)
            if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                return self.inference_image(input, output, prompt)
            elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                return self.inference_video(input, output, prompt)
            else:
                raise ValueError(f"Unsupported file type: {input_path.suffix}")
            
        except Exception as e:
            print(f"\n❌ Error running SAM2 inference: {e}")
            return {"status": "error", "error": str(e)}
            
    def inference_image(self, input, output=None, prompt=None):
        """
        Run inference on an image with the SAM2 model.
        
        Args:
            input (str): Path to input image
            output (str, optional): Path to save the output image
            prompt (dict, optional): Prompt for the model (points, boxes, etc.)
            
        Returns:
            dict: Results from the segmentation
        """
        try:
            print("\n🚀 Starting Exla SAM2 Server Image Inference Pipeline\n")
            
            result = self._process_image(input, output, prompt)
            
            print(f"\n✨ SAM2 Image Inference Summary:")
            print(f"  - Input: {input}")
            if output:
                print(f"  - Output: {output}")
            if "masks" in result:
                print(f"  - Masks generated: {len(result['masks'])}")
            
            return result
            
        except Exception as e:
            print(f"\n❌ Error running SAM2 image inference: {e}")
            return {"status": "error", "error": str(e)}
            
    def inference_video(self, input, output=None, prompt=None):
        """
        Run inference on a video with the SAM2 model.
        
        Args:
            input (str): Path to input video
            output (str, optional): Path to save the output video
            prompt (dict, optional): Prompt for the model (points, boxes, etc.)
            
        Returns:
            dict: Results from the segmentation
        """
        try:
            import cv2
            import numpy as np
            import tempfile
            
            print("\n🚀 Starting Exla SAM2 Server Video Inference Pipeline\n")
            
            # Open the video
            cap = cv2.VideoCapture(input)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {input}")
                
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
            
            # Create output video writer if output is specified
            if output:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output, fourcc, fps, (width, height))
            
            # Process each frame
            frame_count = 0
            all_masks = []
            
            print("Processing video frames...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                print(f"Processing frame {frame_count}/{total_frames}")
                
                # Save frame to temporary file
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp:
                    temp_path = temp.name
                    cv2.imwrite(temp_path, frame)
                
                # Process the frame
                result = self._process_image(temp_path, None, prompt)
                
                # Remove temporary file
                os.unlink(temp_path)
                
                # Store masks
                if "masks" in result:
                    all_masks.append(result["masks"])
                
                # Create visualization if output is specified
                if output and "masks" in result:
                    # Convert masks to numpy arrays
                    masks = [np.array(mask) for mask in result["masks"]]
                    
                    # Create a visualization of the masks
                    mask_image = np.zeros_like(frame)
                    for i, mask in enumerate(masks):
                        color = np.random.randint(0, 255, size=3)
                        mask_image[mask] = color
                    
                    # Blend with original frame
                    result_frame = cv2.addWeighted(frame, 0.7, mask_image, 0.3, 0)
                    
                    # Write to output video
                    out.write(result_frame)
            
            # Release resources
            cap.release()
            if output:
                out.release()
                
            print(f"\n✨ SAM2 Video Inference Summary:")
            print(f"  - Input: {input}")
            if output:
                print(f"  - Output: {output}")
            print(f"  - Frames processed: {frame_count}")
            if all_masks:
                print(f"  - Average masks per frame: {sum(len(m) for m in all_masks) / len(all_masks) if all_masks else 0:.2f}")
            
            return {
                "status": "success",
                "frames_processed": frame_count,
                "masks": all_masks
            }
            
        except Exception as e:
            print(f"\n❌ Error running SAM2 video inference: {e}")
            return {"status": "error", "error": str(e)}
            
    def __del__(self):
        """Cleanup when the object is destroyed."""
        try:
            print("\nCleaning up SAM2 server...")
            subprocess.run(["docker", "rm", "-f", self._container_name], 
                          stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            print("SAM2 server stopped")
        except:
            pass 