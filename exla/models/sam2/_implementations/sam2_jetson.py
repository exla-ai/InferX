from ._base import SAM2_Base
import os
import subprocess
import time
import sys
import threading
import itertools
import json
import tempfile
import io
import requests
from pathlib import Path
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"[BENCHMARK] Starting {func.__name__}...")
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"[BENCHMARK] {func.__name__} completed in {elapsed:.4f} seconds")
        return result
    return wrapper

class ProgressIndicator:
    """
    A simple spinner progress indicator with timing information.
    """
    def __init__(self, message):
        self.message = message
        self.start_time = time.time()
        self._spinner_cycle = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
        self._stop_event = threading.Event()
        self._thread = None
        
    def _get_elapsed(self):
        return f"{time.time() - self.start_time:.1f}s"
        
    def _animate(self):
        while not self._stop_event.is_set():
            spinner = next(self._spinner_cycle)
            sys.stdout.write("\r" + " " * 100)  # Clear line
            sys.stdout.write(f"\r{spinner} [{self._get_elapsed()}] {self.message}")
            sys.stdout.flush()
            time.sleep(0.1)
    
    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._animate)
        self._thread.daemon = True
        self._thread.start()
        return self
        
    def stop(self, success=True, final_message=None):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        
        symbol = "✓" if success else "✗"
        message = final_message or self.message
        elapsed = self._get_elapsed()
        
        sys.stdout.write("\r" + " " * 100)  # Clear line
        sys.stdout.write(f"\r{symbol} [{elapsed}] {message}\n")
        sys.stdout.flush()
        
        return elapsed
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        self.stop(success=success)

class SAM2_Jetson(SAM2_Base):
    """
    SAM2 implementation optimized for NVIDIA Jetson platforms using TensorRT.
    Supports both local model inference and server-based inference.
    """
    def __init__(self, model_name="sam2_b", server_url="http://localhost:8000/predict"):
        """
        Initialize the SAM2 model for Jetson devices.
        
        Args:
            model_name (str): Name of the SAM2 model to use
            server_url (str): URL of the SAM2 server
        """
        super().__init__()
        
        # Set model name and checkpoint name based on input
        if model_name == "sam2_b" or model_name == "b":
            self.model_name = "sam2_b"
            self.checkpoint_name = "sam2_b.pth"
        elif model_name == "sam2_l" or model_name == "l":
            self.model_name = "sam2_l"
            self.checkpoint_name = "sam2_l.pth"
        elif model_name == "sam2_h" or model_name == "h":
            self.model_name = "sam2_h"
            self.checkpoint_name = "sam2_h.pth"
        else:
            self.model_name = "sam2_b"
            self.checkpoint_name = "sam2_b.pth"
            
        self.server_url = server_url
        self.model = None
        self.predictor = None
        self.sam = None
        
        # Create cache directory for model downloads
        self.cache_dir = Path.home() / ".cache" / "exla" / "sam2"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if we're using local model or server
        self.use_server = True
        try:
            # Try to connect to server
            response = requests.get(self.server_url.replace("/predict", ""), timeout=1)
        except Exception as e:
            # If server not available, use local model
            self.use_server = False
            self._install_dependencies()
            self._load_model()
        
    def _install_dependencies(self):
        """Install required dependencies for SAM2 with TensorRT"""
        with ProgressIndicator("Installing SAM2 TensorRT dependencies") as progress:
            try:
                # Check if dependencies are already installed
                try:
                    import tensorrt
                    import torch2trt
                    progress.stop(final_message="SAM2 TensorRT dependencies already installed")
                    return
                except ImportError:
                    pass
                
                # Install required dependencies
                subprocess.run([
                    "uv", "pip", "install", "torch", "torchvision", "opencv-python", "matplotlib"
                ], check=True)
                
                # Install SAM2 dependencies
                subprocess.run([
                    "uv", "pip", "install", "git+https://github.com/facebookresearch/segment-anything-2.git"
                ], check=True)
                
                # Install TensorRT dependencies
                subprocess.run([
                    "uv", "pip", "install", "nvidia-tensorrt"
                ], check=True)
                
                # Install torch2trt
                subprocess.run([
                    "uv", "pip", "install", "git+https://github.com/NVIDIA-AI-IOT/torch2trt"
                ], check=True)
                
                progress.stop(final_message="Successfully installed SAM2 TensorRT dependencies")
            except Exception as e:
                progress.stop(success=False, final_message=f"Failed to install dependencies: {str(e)}")
                raise
                
    def _load_model(self):
        """Load the SAM2 model with TensorRT optimizations"""
        try:
            # Import the actual SAM2 model
            try:
                from segment_anything_2 import sam2_model_registry, Sam2Predictor
            except ImportError:
                subprocess.run([
                    "uv", "pip", "install", "git+https://github.com/facebookresearch/segment-anything-2.git"
                ], check=True)
                from segment_anything_2 import sam2_model_registry, Sam2Predictor
            
            # Check if model exists in cache
            checkpoint_path = self.cache_dir / f"{self.checkpoint_name}"
            
            if not checkpoint_path.exists():
                # Create a fallback path to check in the Docker mount location
                docker_path = Path("/tmp/nv_jetson_model/sam2") / f"{self.checkpoint_name}"
                if docker_path.exists():
                    checkpoint_path = docker_path
                else:
                    raise RuntimeError(f"Model file not found. Please download it to {self.cache_dir} or {docker_path}")
            
            # Load model
            # SAM2 model registry uses different keys than the original SAM
            # Convert model name to registry key if needed
            model_registry_key = self.model_name.replace("sam2_", "")  # Convert "sam2_b" to "b"
            self.sam = sam2_model_registry[model_registry_key](checkpoint=str(checkpoint_path))
            self.sam.to(device="cuda")
            
            # Create predictor
            self.predictor = Sam2Predictor(self.sam)
            
        except Exception as e:
            raise RuntimeError(f"Error loading SAM2 model: {str(e)}")
    
    def show_mask(self, mask, ax, random_color=False, borders=True):
        """
        Display a mask on a matplotlib axis.
        
        Args:
            mask: Binary mask to display
            ax: Matplotlib axis to display on
            random_color: Whether to use a random color
            borders: Whether to draw borders around the mask
        """
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            # Vibrant blue with higher opacity
            color = np.array([0.0, 0.447, 1.0, 0.6])  # Strong blue with 0.6 opacity
        
        h, w = mask.shape
        mask_image = np.zeros((h, w, 4), dtype=np.float32)
        mask_image[mask] = color
        
        if borders:
            # Convert mask to uint8 for contour detection
            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create a separate image for the contours
            contour_image = np.zeros((h, w, 4), dtype=np.float32)
            # Draw white contours with full opacity
            cv2.drawContours(contour_image, contours, -1, (1, 1, 1, 1), thickness=2)
            
            # Combine the mask and contours
            mask_image = np.clip(mask_image + contour_image, 0, 1)
        
        ax.imshow(mask_image)
    
    def show_points(self, coords, labels, ax, marker_size=375):
        """
        Display points on a matplotlib axis.
        
        Args:
            coords: Point coordinates
            labels: Point labels (1=foreground, 0=background)
            ax: Matplotlib axis to display on
            marker_size: Size of the markers
        """
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(
            pos_points[:, 0], pos_points[:, 1],
            color='lime', marker='*',  # Brighter green color
            s=marker_size, edgecolor='white',
            linewidth=2.0,  # Thicker white edge
            zorder=2  # Ensure points are drawn on top
        )
        ax.scatter(
            neg_points[:, 0], neg_points[:, 1],
            color='red', marker='*',
            s=marker_size, edgecolor='white',
            linewidth=2.0,
            zorder=2
        )
    
    def resize_mask(self, mask, target_size):
        """
        Resize a binary mask to the target size.
        
        Args:
            mask: Binary mask to resize
            target_size: Target size (width, height)
            
        Returns:
            Resized binary mask
        """
        mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_pil = mask_pil.resize(target_size, Image.NEAREST)
        return np.array(mask_pil) > 127
    
    def predict_masks_from_server(self, image_path):
        """
        Send image to server for prediction and process the response.
        
        Args:
            image_path: Path to the input image
        
        Returns:
            Tuple of (masks, scores, point_coords, point_labels)
        """
        # Read and verify image
        image = Image.open(image_path).convert("RGB")
        if image.size != (1800, 1200):
            image = image.resize((1800, 1200))
        
        # Prepare image for sending
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Send request to server
        files = {'file': ('image.png', img_byte_arr, 'image/png')}
        response = requests.post(self.server_url, files=files)
        
        if response.status_code != 200:
            raise Exception(f"Error from server: {response.text}")
        
        # Process response
        result = response.json()
        
        # Extract masks and scores
        masks = np.array(result['low_res_masks'])
        scores = np.array(result['iou_predictions'])
        
        # Convert masks to boolean
        masks = (masks > 0.0)
        
        # Sort by scores
        sorted_indices = np.argsort(scores[0])[::-1]
        masks = masks[0][sorted_indices]
        scores = scores[0][sorted_indices]
        
        # Default point coordinates (same as server's fixed point)
        point_coords = torch.tensor([[500, 375]])
        point_labels = torch.tensor([1])
        
        return masks, scores, point_coords, point_labels
    
    def visualize_masks(self, image, masks, scores, point_coords, point_labels, output_dir=None, prefix=""):
        """
        Visualize and save masks overlaid on the original image.
        
        Args:
            image: Input image (PIL Image or numpy array)
            masks: Binary masks
            scores: Confidence scores
            point_coords: Point coordinates
            point_labels: Point labels
            output_dir: Directory to save visualizations (optional)
            prefix: Prefix for output filenames
        """
        # Convert image to PIL if it's a numpy array
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
            image = np.array(image_pil)
        
        # Get image dimensions
        image_h, image_w = image.shape[:2]
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Determine output path
        base_path = output_dir if output_dir else "."
        
        # Process each mask
        for i, (mask, score) in enumerate(zip(masks, scores)):
            # Resize mask to match image dimensions if needed
            if mask.shape[0] != image_h or mask.shape[1] != image_w:
                mask = self.resize_mask(mask, (image_w, image_h))
            
            # Create direct overlay with OpenCV
            mask_color = np.zeros_like(image, dtype=np.uint8)
            mask_color[mask] = [0, 114, 255]  # BGR format for OpenCV
            
            # Create a standalone visualization
            overlay = image.copy()
            cv2.addWeighted(mask_color, 0.6, overlay, 0.4, 0, overlay)
            
            # Save the direct overlay - remove underscore from beginning of filename
            filename_prefix = prefix + "direct_overlay" if prefix else "direct_overlay"
            direct_overlay_path = os.path.join(base_path, f"{filename_prefix}_{i+1}.png")
            cv2.imwrite(direct_overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
            # Create matplotlib visualization
            plt.figure(figsize=(10, 10))
            plt.imshow(image_pil)
            self.show_mask(mask, plt.gca(), random_color=(i > 0))
            
            # Add points if provided
            if point_coords is not None and point_labels is not None:
                self.show_points(point_coords, point_labels, plt.gca())
            
            # Format title with prefix (if any)
            title_prefix = f"{prefix} " if prefix else ""
            plt.title(f"{title_prefix}Mask {i + 1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            
            # Save matplotlib visualization - remove underscore from beginning of filename
            output_filename = f"{prefix}output_mask_{i+1}.png" if prefix else f"output_mask_{i+1}.png"
            output_path = os.path.join(base_path, output_filename)
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=200)
            plt.close()
    
    def inference_image(self, input_path, output_path=None, prompt=None, show_visualization=False):
        """
        Run inference on a single image.
        
        Args:
            input_path (str): Path to input image
            output_path (str, optional): Path to save visualization
            prompt (dict, optional): Prompt containing points or box
            show_visualization (bool, optional): Whether to show visualization masks
            
        Returns:
            dict: Results containing masks and scores
        """
        try:
            if self.use_server:
                # Use server for inference
                masks, scores, point_coords, point_labels = self.predict_masks_from_server(input_path)
            else:
                # Use local model for inference
                # Load and preprocess image
                image = cv2.imread(input_path)
                if image is None:
                    raise ValueError(f"Could not read image: {input_path}")
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Set image in predictor
                if self.predictor is None:
                    raise ValueError("SAM2 predictor is not available. Please download the model files.")
                    
                self.predictor.set_image(image)
                
                # Process based on prompt type
                if prompt is None:
                    # Default point in the center of the image
                    h, w = image.shape[:2]
                    default_point = np.array([[w//2, h//2]])
                    default_label = np.array([1])
                    masks, scores, logits = self.predictor.predict(
                        point_coords=default_point,
                        point_labels=default_label,
                        multimask_output=True
                    )
                    # Store the default point in prompt for visualization
                    point_coords = torch.tensor(default_point)
                    point_labels = torch.tensor(default_label)
                else:
                    if "points" in prompt:
                        input_points = np.array(prompt["points"])
                        input_labels = np.array(prompt.get("labels", [1] * len(prompt["points"])))
                        masks, scores, logits = self.predictor.predict(
                            point_coords=input_points,
                            point_labels=input_labels,
                            multimask_output=True
                        )
                        point_coords = torch.tensor(input_points)
                        point_labels = torch.tensor(input_labels)
                    elif "box" in prompt:
                        input_box = np.array(prompt["box"])
                        masks, scores, logits = self.predictor.predict(
                            box=input_box[None, :],
                            multimask_output=True
                        )
                        # No points for box prompt
                        point_coords = None
                        point_labels = None
                    else:
                        raise ValueError("Invalid prompt format. Must contain 'points' or 'box'.")
            
            # Create visualizations if output path is provided and show_visualization is True
            if output_path and show_visualization:
                # Determine if output_path is a directory
                if os.path.isdir(output_path) or not os.path.splitext(output_path)[1]:
                    # It's a directory
                    output_dir = output_path
                    prefix = ""
                else:
                    # It's a file, extract directory and use filename as prefix
                    output_dir = os.path.dirname(output_path)
                    if not output_dir:
                        output_dir = "."
                    prefix = os.path.splitext(os.path.basename(output_path))[0]
                
                # Load image for visualization if using server mode
                if self.use_server:
                    image = Image.open(input_path).convert("RGB")
                
                print(f"BEFORE VISUALIZE MASKS {time.time()}")
                # Generate visualizations
                self.visualize_masks(
                    image=image,
                    masks=masks,
                    scores=scores,
                    point_coords=point_coords,
                    point_labels=point_labels,
                    output_dir=output_dir,
                    prefix=prefix
                )

                print(f"AFTER VISUALIZE MASKS {time.time()}")             
                # Also create a simple overlay for backward compatibility
                if not os.path.isdir(output_path):
                    if self.use_server:
                        image_np = np.array(image)
                    else:
                        image_np = image
                        
                    # Ensure masks are properly sized for the image
                    h, w = image_np.shape[:2]
                    mask_overlay = np.zeros_like(image_np)
                    
                    for i, mask in enumerate(masks):
                        # Resize mask if needed
                        if mask.shape[0] != h or mask.shape[1] != w:
                            mask = self.resize_mask(mask, (w, h))
                            
                        color = np.random.randint(0, 255, size=3)
                        mask_overlay[mask] = color
                    
                    # Blend with original image
                    result = cv2.addWeighted(image_np, 0.7, mask_overlay, 0.3, 0)
                    
                    # Save output
                    cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            
            # Convert masks to list for JSON serialization
            masks_list = masks.tolist() if isinstance(masks, np.ndarray) else masks
            scores_list = scores.tolist() if isinstance(scores, np.ndarray) else scores
            
            return {
                "status": "success",
                "masks": masks_list,
                "scores": scores_list
            }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    

    def inference(self, input, output=None, prompt=None):
        """
        Run inference with the SAM2 model.
        
        Args:
            input (str or int): Path to input image, video file, or camera index/URL
            output (str, optional): Path to save the output
            prompt (dict, optional): Prompt for the model with one of the following formats:
                - Point prompt: {"points": [[x1, y1], ...], "labels": [1, ...]}
                - Box prompt: {"box": [x1, y1, x2, y2]}
                - Mask prompt: {"mask": binary_mask_array}
                - Text prompt: {"text": "description of object"}
            
        Returns:
            dict: Results from the segmentation including:
                - status: "success" or "error"
                - masks: List of binary masks
                - scores: Confidence scores for each mask
                - processing_time: Time taken for inference (if benchmarking)
                - error: Error message (if status is "error")
        """
        # Start timing for benchmarking
        start_time = time.time()
        
        try:
            # Detect input type (camera, video file, or image)
            is_camera = isinstance(input, int) or (isinstance(input, str) and input.startswith(('rtsp://', 'http://', 'https://', '/dev/')))
            is_video_file = isinstance(input, str) and input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'))
            
            # Handle camera input
            if is_camera:
                if self.use_server:
                    return {
                        "status": "error",
                        "error": "Camera processing is not supported in server mode. Please use a local model.",
                        "processing_time": time.time() - start_time
                    }
                # Use camera_source parameter name for inference_camera
                return self.inference_camera(camera_source=input, output=output, prompt=prompt)
            
            # Handle video file
            elif is_video_file:
                if self.use_server:
                    print("⚠️ Warning: Video processing in server mode is limited. Using frame-by-frame processing.")
                    # Use named parameters for process_video_in_server_mode
                    return self.process_video_in_server_mode(
                        video_path=input, 
                        output_path=output, 
                        prompt=prompt, 
                        start_time=start_time
                    )
                else:
                    # Use video_path and output_path parameter names for simulate_camera_with_video
                    return self.simulate_camera_with_video(
                        video_path=input, 
                        output_path=output, 
                        display=False, 
                        center_point=prompt
                    )
            
            # Handle image file
            else:
                # Use input_path and output_path to match the method signature of inference_image
                result = self.inference_image(input_path=input, output_path=output, prompt=prompt)
                result['processing_time'] = time.time() - start_time
                return result
                
        except Exception as e:
            print(f"❌ Error during inference: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    @timing_decorator
    def process_video_in_server_mode(self, video_path, output_path, prompt, start_time):
        """
        Process a video file in server mode by extracting frames, processing them individually,
        and combining them into a video output.
        
        Args:
            video_path: Path to the input video
            output_path: Path to save the output
            prompt: Prompt for segmentation
            start_time: Start time for benchmarking
            
        Returns:
            dict: Results from the segmentation
        """
        try:
            print(f"[BENCHMARK] Starting video processing in server mode: {video_path}")
            
            # Check if output_path is a directory
            setup_start = time.time()
            if os.path.isdir(output_path):
                output_video = os.path.join(output_path, "segmented_video.mp4")
                # Ensure the directory exists
                os.makedirs(output_path, exist_ok=True)
            else:
                output_video = output_path
                # Ensure the parent directory exists
                output_dir = os.path.dirname(output_video)
                if output_dir:  # Only create if there's a directory component
                    os.makedirs(output_dir, exist_ok=True)
            print(f"[BENCHMARK] Output setup completed in {time.time() - setup_start:.4f} seconds")
            
            # Open the video file
            video_open_start = time.time()
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
                
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"[BENCHMARK] Video open and property reading completed in {time.time() - video_open_start:.4f} seconds")
            
            # Setup output video writer
            writer_start = time.time()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
            print(f"[BENCHMARK] Video writer setup completed in {time.time() - writer_start:.4f} seconds")
            
            # Process frames
            frame_count = 0
            processing_times = []
            
            print(f"[BENCHMARK] Processing video with {total_frames} frames at {fps} FPS")
            print(f"[BENCHMARK] Frame dimensions: {width}x{height}")
            
            # Flag to track if we've shown the resize message
            resize_message_shown = False
            
            # Create a temporary directory for frames
            temp_dir_start = time.time()
            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"[BENCHMARK] Temporary directory creation completed in {time.time() - temp_dir_start:.4f} seconds")
                
                while True:
                    frame_start = time.time()
                    # Read frame
                    read_start = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        break
                    read_time = time.time() - read_start
                    
                    # Save frame as temporary image
                    save_start = time.time()
                    frame_path = os.path.join(temp_dir, f"frame_{frame_count:06d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    save_time = time.time() - save_start
                    
                    # Process frame with SAM2
                    frame_start_time = time.time()
                    
                    # Use the provided prompt or default to center point
                    if prompt is None:
                        # Default point in the center of the image
                        h, w = frame.shape[:2]
                        frame_prompt = {
                            "points": [[w//2, h//2]],
                            "labels": [1]
                        }
                    else:
                        frame_prompt = prompt
                    
                    # Process the frame - use input_path instead of input to match the method signature
                    inference_start = time.time()
                    frame_result = self.inference_image(
                        input_path=frame_path,
                        output_path=None,  # Don't save intermediate results
                        prompt=frame_prompt
                    )
                    inference_time = time.time() - inference_start
                    
                    # Check if processing was successful
                    visualization_start = time.time()
                    if frame_result.get("status") == "success" and "masks" in frame_result:
                        # Get the best mask
                        masks = frame_result["masks"]
                        scores = frame_result["scores"]
                        
                        if len(masks) > 0:
                            # Use the highest scoring mask
                            best_mask_idx = np.argmax(scores)
                            mask = np.array(masks[best_mask_idx])
                            
                            # Show resize message only for the first frame
                            if not resize_message_shown:
                                print(f"[BENCHMARK] Resizing mask from {mask.shape} to {(height, width)}")
                                resize_message_shown = True
                            
                            # Convert mask to PIL Image for resizing
                            from PIL import Image
                            mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
                            mask_pil = mask_pil.resize((width, height), Image.NEAREST)
                            
                            # Convert back to numpy array and ensure it's boolean
                            mask_resized = np.array(mask_pil) > 127
                            
                            # Create visualization
                            mask_overlay = np.zeros_like(frame)
                            mask_overlay[mask_resized] = [0, 114, 255]  # BGR format for OpenCV
                            
                            # Blend with original frame
                            result_frame = cv2.addWeighted(frame, 0.7, mask_overlay, 0.3, 0)
                            
                            # Add frame number and processing time
                            process_time = time.time() - frame_start_time
                            processing_times.append(process_time)
                            avg_time = sum(processing_times) / len(processing_times)
                            
                            cv2.putText(result_frame, f"Frame: {frame_count} | Time: {process_time:.3f}s | Avg: {avg_time:.3f}s", 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # Write to output video
                            video_writer.write(result_frame)
                        else:
                            # No masks found, use original frame
                            video_writer.write(frame)
                    else:
                        # Processing failed, use original frame
                        video_writer.write(frame)
                    visualization_time = time.time() - visualization_start
                    
                    frame_count += 1
                    frame_total_time = time.time() - frame_start
                    
                    # Print progress every 10 frames
                    if frame_count % 10 == 0:
                        progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
                        print(f"[BENCHMARK] Progress: {progress:.1f}% ({frame_count}/{total_frames}) | Avg time: {avg_time:.3f}s per frame")
                        print(f"[BENCHMARK] Frame {frame_count} breakdown - Read: {read_time:.4f}s, Save: {save_time:.4f}s, Inference: {inference_time:.4f}s, Visualization: {visualization_time:.4f}s, Total: {frame_total_time:.4f}s")
            
            # Clean up
            cleanup_start = time.time()
            cap.release()
            video_writer.release()
            print(f"[BENCHMARK] Cleanup completed in {time.time() - cleanup_start:.4f} seconds")
            
            # Calculate total processing time
            total_time = time.time() - start_time
            avg_frame_time = sum(processing_times) / len(processing_times) if processing_times else 0
            
            print(f"[BENCHMARK] Video processing complete: {frame_count} frames in {total_time:.3f}s")
            print(f"[BENCHMARK] Average processing time: {avg_frame_time:.3f}s per frame")
            print(f"[BENCHMARK] Output saved to: {output_video}")
            
            return {
                "status": "success",
                "frames_processed": frame_count,
                "average_time": avg_frame_time,
                "total_time": total_time,
                "output_path": output_video,
                "processing_time": total_time
            }
            
        except Exception as e:
            print(f"❌ Error during video processing: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    @timing_decorator
    def get_model(self):
        """
        Get direct access to the underlying SAM2 model.
        
        Returns:
            The SAM2 model object for direct manipulation
        """
        if self.use_server:
            raise RuntimeError("Direct model access is not available when using server mode. Please initialize with use_server=False.")
        
        if self.sam is None:
            raise RuntimeError("Model not loaded. Please ensure the model is loaded before accessing it.")
            
        return self.sam
        
    @timing_decorator
    def get_predictor(self):
        """
        Get direct access to the SAM2 predictor.
        
        Returns:
            The SAM2 predictor object for direct manipulation
        """
        if self.use_server:
            raise RuntimeError("Direct predictor access is not available when using server mode. Please initialize with use_server=False.")
        
        if self.predictor is None:
            raise RuntimeError("Predictor not loaded. Please ensure the model is loaded before accessing it.")
            
        return self.predictor
        
    @timing_decorator
    def simulate_camera_with_video(self, video_path, output_path=None, display=True, center_point=None, max_frames=None):
        """
        Simulate camera input using a video file and run SAM2 segmentation on each frame.
        
        Args:
            video_path (str): Path to the input video file
            output_path (str, optional): Path to save the output video
            display (bool): Whether to display the output in a window
            center_point (dict, optional): Point to use for segmentation (default: center of frame)
            max_frames (int, optional): Maximum number of frames to process
            
        Returns:
            dict: Results from the segmentation including processing statistics
        """
        print(f"[BENCHMARK] Starting camera simulation with video: {video_path}")
        
        if self.use_server:
            print("⚠️ Camera simulation is not supported in server mode. Please use local model.")
            return {"status": "error", "error": "Camera simulation not supported in server mode"}
            
        if self.predictor is None:
            print("❌ Error: SAM2 predictor is not available. Cannot process video.")
            print("This may be due to missing model files or initialization issues.")
            return {"status": "error", "error": "SAM2 predictor not available"}
        
        try:
            # Open the video file
            video_open_start = time.time()
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
                
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"[BENCHMARK] Video open completed in {time.time() - video_open_start:.4f} seconds")
            print(f"[BENCHMARK] Video properties: {width}x{height} at {fps} FPS")
            
            # Setup output video writer if needed
            setup_start = time.time()
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
            print(f"[BENCHMARK] Setup completed in {time.time() - setup_start:.4f} seconds")
            
            # Define tracking point (default: center of the frame)
            if center_point is None:
                center_point = {
                    "points": [[width//2, height//2]],
                    "labels": [1]  # Foreground point
                }
            
            # Process frames
            frame_count = 0
            processing_times = []
            avg_time = 0
            results = []
            
            print(f"[BENCHMARK] Starting frame processing with SAM2")
            print("[BENCHMARK] Press 'q' to quit.")
            
            while True:
                frame_start = time.time()
                
                # Read frame
                read_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                read_time = time.time() - read_start
                
                # Check if we've reached max frames
                if max_frames is not None and frame_count >= max_frames:
                    break
                
                try:
                    # Convert frame to RGB for SAM2
                    convert_start = time.time()
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    convert_time = time.time() - convert_start
                    
                    # Set image in predictor
                    set_image_start = time.time()
                    self.predictor.set_image(frame_rgb)
                    set_image_time = time.time() - set_image_start
                    
                    # Process with center point
                    point_coords = np.array(center_point["points"])
                    point_labels = np.array(center_point["labels"])
                    
                    predict_start = time.time()
                    masks, scores, logits = self.predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=True
                    )
                    predict_time = time.time() - predict_start
                    
                    # Use the highest scoring mask
                    best_mask_idx = np.argmax(scores)
                    mask = masks[best_mask_idx]
                    score = scores[best_mask_idx]
                    
                    # Create visualization
                    vis_start = time.time()
                    mask_overlay = np.zeros_like(frame)
                    mask_overlay[mask] = [0, 114, 255]  # BGR format for OpenCV
                    
                    # Draw the center point
                    for coord, label in zip(point_coords, point_labels):
                        color = (0, 255, 0) if label == 1 else (0, 0, 255)
                        cv2.drawMarker(frame, (int(coord[0]), int(coord[1])), color, 
                                      markerType=cv2.MARKER_STAR, markerSize=20, thickness=2)
                    
                    # Blend with original frame
                    result_frame = cv2.addWeighted(frame, 0.7, mask_overlay, 0.3, 0)
                    
                    # Add processing time text
                    process_time = time.time() - start_time
                    processing_times.append(process_time)
                    avg_time = sum(processing_times) / len(processing_times)
                    cv2.putText(result_frame, f"SAM2 | Frame: {frame_count} | Time: {process_time:.3f}s | Avg: {avg_time:.3f}s", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    vis_time = time.time() - vis_start
                    
                    # Store result
                    results.append({
                        "frame": frame_count,
                        "score": float(score),
                        "process_time": process_time
                    })
                    
                    # Write to output video if needed
                    write_start = time.time()
                    if video_writer:
                        video_writer.write(result_frame)
                    write_time = time.time() - write_start
                    
                    # Display if needed
                    display_start = time.time()
                    if display:
                        cv2.imshow("SAM2 Segmentation", result_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                    display_time = time.time() - display_start
                
                except Exception as e:
                    print(f"❌ Error processing frame {frame_count}: {str(e)}")
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
                    
                    # Store error result
                    results.append({
                        "frame": frame_count,
                        "error": str(e)
                    })
                
                frame_count += 1
                frame_total_time = time.time() - frame_start
                
                # Print progress every 10 frames
                if frame_count % 10 == 0:
                    print(f"[BENCHMARK] Processed {frame_count} frames. Average time: {avg_time:.3f}s per frame")
                    print(f"[BENCHMARK] Frame {frame_count} breakdown - Read: {read_time:.4f}s, Convert: {convert_time:.4f}s, Set Image: {set_image_time:.4f}s, Predict: {predict_time:.4f}s, Visualize: {vis_time:.4f}s, Write: {write_time:.4f}s, Display: {display_time:.4f}s, Total: {frame_total_time:.4f}s")
            
            # Clean up
            cleanup_start = time.time()
            cap.release()
            if video_writer:
                video_writer.release()
            if display:
                cv2.destroyAllWindows()
            print(f"[BENCHMARK] Cleanup completed in {time.time() - cleanup_start:.4f} seconds")
            
            print(f"[BENCHMARK] Processing complete: {frame_count} frames processed")
            if len(processing_times) > 0:
                print(f"[BENCHMARK] Average processing time: {avg_time:.3f}s per frame")
            
            return {
                "status": "success",
                "frames_processed": frame_count,
                "average_time": avg_time if len(processing_times) > 0 else None,
                "results": results
            }
            
        except Exception as e:
            print(f"❌ Error during video processing: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
        
    @timing_decorator
    def inference_camera(self, camera_source, output=None, prompt=None, max_frames=None, fps=30, display=False):
        """
        Run inference on a camera stream.
        
        Args:
            camera_source: Camera index (0, 1, etc.) or URL (rtsp://, http://, etc.)
            output (str, optional): Path to save output video
            prompt (dict, optional): Prompt for the model (points, boxes, etc.)
            max_frames (int, optional): Maximum number of frames to process
            fps (int): Frames per second for output video
            display (bool): Whether to display the output in a window
            
        Returns:
            dict: Results from the segmentation
        """
        print(f"[BENCHMARK] Starting camera inference from source: {camera_source}")
        
        if self.use_server:
            print("⚠️ Camera stream inference is not supported in server mode. Please use local model.")
            return {"status": "error", "error": "Camera stream inference not supported in server mode"}
            
        try:
            # Open camera
            camera_open_start = time.time()
            cap = cv2.VideoCapture(camera_source)
            if not cap.isOpened():
                raise ValueError(f"Could not open camera source: {camera_source}")
                
            # Get camera properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[BENCHMARK] Camera open completed in {time.time() - camera_open_start:.4f} seconds")
            print(f"[BENCHMARK] Camera resolution: {width}x{height}")
            
            # Setup output video writer if needed
            setup_start = time.time()
            video_writer = None
            if output:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output, fourcc, fps, (width, height))
            
            # Process prompt
            if prompt is None:
                # Default point in the center of the image
                default_point = np.array([[width//2, height//2]])
                default_label = np.array([1])
                point_coords = default_point
                point_labels = default_label
            else:
                if "points" in prompt:
                    point_coords = np.array(prompt["points"])
                    point_labels = np.array(prompt.get("labels", [1] * len(prompt["points"])))
                elif "box" in prompt:
                    # For box prompt, we'll use it directly in the predict call
                    point_coords = None
                    point_labels = None
                else:
                    raise ValueError("Invalid prompt format. Must contain 'points' or 'box'.")
            
            # Setup display window if needed
            if display:
                cv2.namedWindow("SAM2 Segmentation", cv2.WINDOW_NORMAL)
            print(f"[BENCHMARK] Setup completed in {time.time() - setup_start:.4f} seconds")
            
            frame_count = 0
            results = []
            processing_times = []
            
            print("[BENCHMARK] Starting camera stream inference. Press 'q' to quit.")
            
            while True:
                frame_start = time.time()
                
                # Read frame
                read_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                read_time = time.time() - read_start
                    
                # Check if we've reached max frames
                if max_frames is not None and frame_count >= max_frames:
                    break
                
                # Convert frame to RGB for SAM
                convert_start = time.time()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                convert_time = time.time() - convert_start
                
                # Set image in predictor
                set_image_start = time.time()
                self.predictor.set_image(frame_rgb)
                set_image_time = time.time() - set_image_start
                
                # Process based on prompt type
                predict_start = time.time()
                if prompt is None or "points" in prompt:
                    masks, scores, logits = self.predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=True
                    )
                elif "box" in prompt:
                    input_box = np.array(prompt["box"])
                    masks, scores, logits = self.predictor.predict(
                        box=input_box[None, :],
                        multimask_output=True
                    )
                predict_time = time.time() - predict_start
                
                # Create visualization
                vis_start = time.time()
                # Use the highest scoring mask
                best_mask_idx = np.argmax(scores)
                mask = masks[best_mask_idx]
                
                # Create mask overlay
                mask_overlay = np.zeros_like(frame)
                mask_overlay[mask] = [0, 114, 255]  # BGR format for OpenCV
                
                # Blend with original frame
                result = cv2.addWeighted(frame, 0.7, mask_overlay, 0.3, 0)
                
                # Draw points if available
                if point_coords is not None:
                    for i, (coord, label) in enumerate(zip(point_coords, point_labels)):
                        color = (0, 255, 0) if label == 1 else (0, 0, 255)  # Green for positive, Red for negative
                        cv2.drawMarker(result, (int(coord[0]), int(coord[1])), color, 
                                      markerType=cv2.MARKER_STAR, markerSize=20, thickness=2)
                
                # Add processing time information
                process_time = time.time() - frame_start
                processing_times.append(process_time)
                avg_time = sum(processing_times) / len(processing_times)
                cv2.putText(result, f"Frame: {frame_count} | Time: {process_time:.3f}s | Avg: {avg_time:.3f}s", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                vis_time = time.time() - vis_start
                
                # Write to output video if needed
                write_start = time.time()
                if video_writer:
                    video_writer.write(result)
                write_time = time.time() - write_start
                
                # Display if needed
                display_start = time.time()
                if display:
                    cv2.imshow("SAM2 Segmentation", result)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                display_time = time.time() - display_start
                
                # Store results
                results.append({
                    "frame": frame_count,
                    "mask": mask.tolist(),
                    "score": float(scores[best_mask_idx])
                })
                
                frame_count += 1
                frame_total_time = time.time() - frame_start
                
                # Print detailed timing information every 10 frames
                if frame_count % 10 == 0:
                    print(f"[BENCHMARK] Processed {frame_count} frames. Average time: {avg_time:.3f}s per frame")
                    print(f"[BENCHMARK] Frame {frame_count} breakdown - Read: {read_time:.4f}s, Convert: {convert_time:.4f}s, Set Image: {set_image_time:.4f}s, Predict: {predict_time:.4f}s, Visualize: {vis_time:.4f}s, Write: {write_time:.4f}s, Display: {display_time:.4f}s, Total: {frame_total_time:.4f}s")
            
            # Clean up
            cleanup_start = time.time()
            cap.release()
            if video_writer:
                video_writer.release()
            if display:
                cv2.destroyAllWindows()
            print(f"[BENCHMARK] Cleanup completed in {time.time() - cleanup_start:.4f} seconds")
            
            print(f"[BENCHMARK] Camera inference complete: {frame_count} frames processed")
            if len(processing_times) > 0:
                print(f"[BENCHMARK] Average processing time: {avg_time:.3f}s per frame")
            
            return {
                "status": "success",
                "frames_processed": frame_count,
                "average_time": avg_time if len(processing_times) > 0 else None,
                "results": results
            }
            
        except Exception as e:
            print(f"❌ Error during camera inference: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            } 