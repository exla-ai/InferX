from ._base import SAM2_Base
import os
import time
import sys
import json
import tempfile
import io
import requests
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
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


class SAM2_Jetson(SAM2_Base):
    """
    SAM2 implementation for server-based inference.
    This implementation assumes the server is always available at the specified URL.
    """
    def __init__(self, server_url="http://localhost:8000/predict"):
        """
        Initialize the SAM2 model for server-based inference.
        
        Args:
            server_url (str): URL of the SAM2 server
        """
        super().__init__()
        self.server_url = server_url
        
        # Verify server connection
        try:
            response = requests.get(self.server_url.replace("/predict", ""), timeout=1)
            print(f"Connected to SAM2 server at {self.server_url}")
        except Exception as e:
            print(f"Warning: Could not connect to SAM2 server at {self.server_url}: {e}")
            print("Inference calls will still be attempted, but may fail if the server is not available.")
    
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
        point_coords = np.array([[500, 375]])
        point_labels = np.array([1])
        
        return masks, scores, point_coords, point_labels
    
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
        Run inference on a single image using the server.
        
        Args:
            input_path (str): Path to input image
            output_path (str, optional): Path to save visualization
            prompt (dict, optional): Prompt containing points or box
            show_visualization (bool, optional): Whether to show visualization masks
            
        Returns:
            dict: Results containing masks and scores
        """
        try:
            # Use server for inference
            masks, scores, point_coords, point_labels = self.predict_masks_from_server(input_path)
            
            # Create visualizations if output path is provided
            if output_path:
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
                
                # Load image for visualization
                image = Image.open(input_path).convert("RGB")
                
                # Generate visualizations if requested
                if show_visualization:
                    self.visualize_masks(
                        image=image,
                        masks=masks,
                        scores=scores,
                        point_coords=point_coords,
                        point_labels=point_labels,
                        output_dir=output_dir,
                        prefix=prefix
                    )
                
                # Create a simple overlay for backward compatibility
                if not os.path.isdir(output_path):
                    image_np = np.array(image)
                    
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

    def inference(self, input, output=None, prompt=None):
        # Start timing for benchmarking
        start_time = time.time()
        try:
            # If input is a NumPy array, convert it to a temporary file.
            if isinstance(input, np.ndarray):
                # Convert from BGR (OpenCV) to RGB (PIL)
                image_pil = Image.fromarray(cv2.cvtColor(input, cv2.COLOR_BGR2RGB))
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                    tmp_file_name = tmp_file.name
                    image_pil.save(tmp_file_name)
                input_path = tmp_file_name
            else:
                input_path = input

            # Detect input type (video file or image)
            is_video_file = isinstance(input_path, str) and input_path.lower().endswith(
                ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv')
            )

            if is_video_file:
                print("⚠️ Warning: Video processing in server mode is limited. Using frame-by-frame processing.")
                return self.process_video_in_server_mode(
                    video_path=input_path, 
                    output_path=output, 
                    prompt=prompt, 
                    start_time=start_time
                )
            else:
                result = self.inference_image(input_path=input_path, output_path=output, prompt=prompt)
                result['processing_time'] = time.time() - start_time
                return result

        except Exception as e:
            print(f"❌ Error during inference: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
