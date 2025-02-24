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
    def __init__(self, model_name="sam2_b", server_url="http://localhost:8000/predict"):
        """
        Initializes SAM2 model on Jetson with TensorRT optimizations.
        
        Args:
            model_name (str): Name of the SAM2 model to use
            server_url (str): URL of the SAM2 prediction server
        """
        super().__init__()
        self.model_name = model_name
        self.server_url = server_url
        self.model = None
        self.predictor = None
        
        # Create cache directory for model downloads
        # Use a directory in the user's home folder instead of /tmp which might have permission issues
        self.cache_dir = Path.home() / ".cache" / "exla" / "sam2"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if we're using local model or server
        self.use_server = True
        try:
            # Try to connect to server
            response = requests.get(self.server_url.replace("/predict", ""), timeout=1)
            print(f"✓ Connected to SAM2 server at {self.server_url}")
        except:
            # If server not available, use local model
            self.use_server = False
            print("Server not available, using local model")
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
                    "uv", "pip", "install", "torch", "torchvision", "opencv-python", "matplotlib", "segment-anything"
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
            from segment_anything import sam_model_registry, SamPredictor
            
            # Check if model exists in cache
            checkpoint_path = self.cache_dir / f"{self.model_name}.pth"
            
            if not checkpoint_path.exists():
                print(f"⚠️ Model not found at {checkpoint_path}.")
                print(f"Please download the SAM2 model file ({self.model_name}.pth) and place it in {self.cache_dir}")
                print("You can download the model from: https://github.com/facebookresearch/segment-anything/tree/main#model-checkpoints")
                
                # Create a fallback path to check in the Docker mount location
                docker_path = Path("/tmp/nv_jetson_model/sam2") / f"{self.model_name}.pth"
                if docker_path.exists():
                    print(f"✓ Found model at Docker mount path: {docker_path}")
                    checkpoint_path = docker_path
                else:
                    raise RuntimeError(f"Model file not found. Please download it to {self.cache_dir} or {docker_path}")
            
            # Load model
            print(f"Loading SAM2 model from: {checkpoint_path}")
            sam = sam_model_registry[self.model_name](checkpoint=str(checkpoint_path))
            sam.to(device="cuda")
            
            # Create predictor
            self.predictor = SamPredictor(sam)
            print(f"✓ Loaded SAM2 model: {self.model_name}")
            
        except Exception as e:
            print(f"❌ Error loading SAM2 model: {str(e)}")
            raise
    
    def show_mask(self, mask, ax, random_color=False, borders=True):
        """Display a mask on a matplotlib axis."""
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
        """Display points on a matplotlib axis."""
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
        """Resize a binary mask to the target size."""
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
            print(f"Resizing image from {image.size} to (1800, 1200)")
            image = image.resize((1800, 1200))
        
        # Prepare image for sending
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Send request to server
        files = {'file': ('image.png', img_byte_arr, 'image/png')}
        print("Sending request to server...")
        response = requests.post(self.server_url, files=files)
        
        if response.status_code != 200:
            raise Exception(f"Error from server: {response.text}")
        
        # Process response
        result = response.json()
        print("Got response from server")
        
        # Extract masks and scores
        masks = np.array(result['low_res_masks'])
        scores = np.array(result['iou_predictions'])
        
        print(f"Raw masks shape: {masks.shape}")
        print(f"Raw scores shape: {scores.shape}")
        
        # Convert masks to boolean
        masks = (masks > 0.0)
        
        # Sort by scores
        sorted_indices = np.argsort(scores[0])[::-1]
        masks = masks[0][sorted_indices]
        scores = scores[0][sorted_indices]
        
        # Default point coordinates (same as server's fixed point)
        point_coords = torch.tensor([[500, 375]])
        point_labels = torch.tensor([1])
        
        print(f"Processed {len(masks)} masks")
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
        
        print(f"Visualizing {len(masks)} masks")
        
        # Process each mask
        for i, (mask, score) in enumerate(zip(masks, scores)):
            # Resize mask to match image dimensions if needed
            if mask.shape[0] != image_h or mask.shape[1] != image_w:
                print(f"Resizing mask from {mask.shape} to ({image_h}, {image_w})")
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
            print(f"Saved direct overlay to {direct_overlay_path}")
            
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
            
            print(f"Saved mask {i+1} to {output_path}")
    
    def inference_image(self, input_path, output_path=None, prompt=None):
        """
        Run inference on a single image.
        
        Args:
            input_path (str): Path to input image
            output_path (str, optional): Path to save visualization
            prompt (dict, optional): Prompt containing points or box
            
        Returns:
            dict: Results containing masks and scores
        """
        try:
            if self.use_server:
                # Use server for inference
                masks, scores, point_coords, point_labels = self.predict_masks_from_server(input_path)
                
                # Create visualizations if output path is provided
                if output_path:
                    # Load image for visualization
                    image = Image.open(input_path).convert("RGB")
                    
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
                        image_np = np.array(image)
                        mask_overlay = np.zeros_like(image_np)
                        for i, mask in enumerate(masks):
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
            else:
                # Use local model for inference
                # Load and preprocess image
                image = cv2.imread(input_path)
                if image is None:
                    raise ValueError(f"Could not read image: {input_path}")
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Set image in predictor
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
                
                # Create visualizations if output path is provided
                if output_path:
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
                        mask_overlay = np.zeros_like(image)
                        for i, mask in enumerate(masks):
                            color = np.random.randint(0, 255, size=3)
                            mask_overlay[mask] = color
                        
                        # Blend with original image
                        result = cv2.addWeighted(image, 0.7, mask_overlay, 0.3, 0)
                        
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
            print(f"❌ Error during inference: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def inference(self, input, output=None, prompt=None):
        """
        Run inference with the SAM2 model.
        
        Args:
            input (str): Path to input image
            output (str, optional): Path to save the output
            prompt (dict, optional): Prompt for the model (points, boxes, etc.)
            
        Returns:
            dict: Results from the segmentation
        """
        return self.inference_image(input, output, prompt) 