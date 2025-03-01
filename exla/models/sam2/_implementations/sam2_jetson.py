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
    
    def visualize_masks(self, image, masks, scores, point_coords, point_labels, output_dir=None, prefix="Torch-TRT"):
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
            
            # Save the direct overlay
            direct_overlay_path = os.path.join(base_path, f"{prefix}_direct_overlay_{i+1}.png")
            cv2.imwrite(direct_overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
            # Create matplotlib visualization
            plt.figure(figsize=(10, 10))
            plt.imshow(image_pil)
            self.show_mask(mask, plt.gca(), random_color=(i > 0))
            
            # Add points if provided
            if point_coords is not None and point_labels is not None:
                self.show_points(point_coords, point_labels, plt.gca())
            
            plt.title(f"{prefix} Mask {i + 1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            
            # Save matplotlib visualization
            output_path = os.path.join(base_path, f"{prefix}_output_mask_{i+1}.png")
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
                    prefix = "Torch-TRT"
                else:
                    # It's a file, extract directory and use filename as prefix
                    output_dir = os.path.dirname(output_path)
                    if not output_dir:
                        output_dir = "."
                    prefix = os.path.splitext(os.path.basename(output_path))[0]
                
                # Load image for visualization if using server mode
                if self.use_server:
                    image = Image.open(input_path).convert("RGB")
                
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
            input (str): Path to input image or camera index/URL
            output (str, optional): Path to save the output
            prompt (dict, optional): Prompt for the model (points, boxes, etc.)
            
        Returns:
            dict: Results from the segmentation
        """
        # Check if input is a camera stream
        if isinstance(input, (int, str)) and (isinstance(input, int) or input.startswith(('rtsp://', 'http://', 'https://', '/dev/'))):
            return self.inference_camera(input, output, prompt)
        else:
            return self.inference_image(input, output, prompt)
            
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
        if self.use_server:
            print("⚠️ Camera stream inference is not supported in server mode. Please use local model.")
            return {"status": "error", "error": "Camera stream inference not supported in server mode"}
            
        try:
            # Open camera
            cap = cv2.VideoCapture(camera_source)
            if not cap.isOpened():
                raise ValueError(f"Could not open camera source: {camera_source}")
                
            # Get camera properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Setup output video writer if needed
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
            
            frame_count = 0
            results = []
            
            print("Starting camera stream inference. Press 'q' to quit.")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Check if we've reached max frames
                if max_frames is not None and frame_count >= max_frames:
                    break
                
                # Convert frame to RGB for SAM
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Set image in predictor
                self.predictor.set_image(frame_rgb)
                
                # Process based on prompt type
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
                
                # Create visualization
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
                
                # Write to output video if needed
                if video_writer:
                    video_writer.write(result)
                
                # Display if needed
                if display:
                    cv2.imshow("SAM2 Segmentation", result)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                
                # Store results
                results.append({
                    "frame": frame_count,
                    "mask": mask.tolist(),
                    "score": float(scores[best_mask_idx])
                })
                
                frame_count += 1
            
            # Clean up
            cap.release()
            if video_writer:
                video_writer.release()
            if display:
                cv2.destroyAllWindows()
            
            return {
                "status": "success",
                "frames_processed": frame_count,
                "results": results
            }
            
        except Exception as e:
            print(f"❌ Error during camera inference: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            } 