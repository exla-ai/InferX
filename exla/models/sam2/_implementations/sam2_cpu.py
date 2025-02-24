from ._base import SAM2_Base
import subprocess
import os
from pathlib import Path

class SAM2_CPU(SAM2_Base):
    def __init__(self, model_name="sam2_b"):
        """
        Initializes a SAM2 model on CPU.
        
        Args:
            model_name (str): Name of the SAM2 model to use
        """
        super().__init__()
        self.model_name = model_name
        self.model = None
        
        # Create cache directory for model downloads
        self.cache_dir = Path.home() / ".cache" / "exla" / "sam2"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Install dependencies
        self._install_dependencies()
        
        # Load model
        self._load_model()
        
    def _install_dependencies(self):
        """Install required dependencies for SAM2"""
        try:
            # Check if dependencies are already installed
            import segment_anything
            print("SAM2 dependencies already installed")
            return
        except ImportError:
            pass
            
        print("Installing SAM2 dependencies...")
        subprocess.run([
            "uv", "pip", "install", "torch", "torchvision", "opencv-python", "matplotlib", "segment-anything"
        ], check=True)
        print("Successfully installed SAM2 dependencies")
        
    def _load_model(self):
        """Load the SAM2 model"""
        try:
            import torch
            from segment_anything import sam_model_registry, SamPredictor
            
            # Download model if not exists
            checkpoint_path = self.cache_dir / f"{self.model_name}.pth"
            if not checkpoint_path.exists():
                print(f"Downloading SAM2 model: {self.model_name}...")
                # This is a placeholder - in a real implementation, you would download the model
                # from the appropriate source
                
            # Load model
            print(f"Loading SAM2 model: {self.model_name}...")
            sam = sam_model_registry[self.model_name](checkpoint=str(checkpoint_path))
            self.model = SamPredictor(sam)
            print("SAM2 model loaded successfully")
            
        except Exception as e:
            print(f"Error loading SAM2 model: {str(e)}")
            raise
    
    def _process_image(self, image_path, output_path=None, prompt=None):
        """
        Process a single image with SAM2.
        
        Args:
            image_path (str): Path to input image
            output_path (str, optional): Path to save the output
            prompt (dict, optional): Prompt for the model
            
        Returns:
            dict: Results from the segmentation
        """
        import cv2
        import numpy as np
        
        # Load image
        print(f"Loading image: {image_path}")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image in the model
        self.model.set_image(image)
        
        # Process prompt
        if prompt is None:
            # If no prompt is provided, use automatic mode
            # This is a placeholder - SAM2 might have different automatic modes
            masks, scores, logits = self.model.predict()
        else:
            # Process based on prompt type
            if "points" in prompt:
                input_points = np.array(prompt["points"])
                input_labels = np.array(prompt.get("labels", [1] * len(prompt["points"])))
                masks, scores, logits = self.model.predict(
                    point_coords=input_points,
                    point_labels=input_labels
                )
            elif "box" in prompt:
                input_box = np.array(prompt["box"])
                masks, scores, logits = self.model.predict(
                    box=input_box
                )
        
        # Save output if specified
        if output_path:
            print(f"Saving output to: {output_path}")
            # Create a visualization of the masks
            mask_image = np.zeros_like(image)
            for i, mask in enumerate(masks):
                color = np.random.randint(0, 255, size=3)
                mask_image[mask] = color
            
            # Blend with original image
            result = cv2.addWeighted(image, 0.7, mask_image, 0.3, 0)
            
            # Convert back to BGR for saving with OpenCV
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, result)
        
        return {
            "status": "success",
            "masks": masks,
            "scores": scores
        }
            
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
            print("\n🚀 Starting Exla SAM2 Inference Pipeline\n")
            
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
            print("\n🚀 Starting Exla SAM2 Image Inference Pipeline\n")
            
            result = self._process_image(input, output, prompt)
            
            print(f"\n✨ SAM2 Image Inference Summary:")
            print(f"  - Input: {input}")
            if output:
                print(f"  - Output: {output}")
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
            
            print("\n🚀 Starting Exla SAM2 Video Inference Pipeline\n")
            
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
                all_masks.append(result['masks'])
                
                # Create visualization if output is specified
                if output:
                    # Create a visualization of the masks
                    mask_image = np.zeros_like(frame)
                    for i, mask in enumerate(result['masks']):
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
            print(f"  - Average masks per frame: {sum(len(m) for m in all_masks) / len(all_masks) if all_masks else 0:.2f}")
            
            return {
                "status": "success",
                "frames_processed": frame_count,
                "masks": all_masks
            }
            
        except Exception as e:
            print(f"\n❌ Error running SAM2 video inference: {e}")
            return {"status": "error", "error": str(e)} 