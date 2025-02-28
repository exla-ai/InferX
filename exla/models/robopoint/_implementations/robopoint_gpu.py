"""
RoboPoint model implementation for GPU.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

from ._base import Robopoint_Base
import torch

class RoboPointGPU(Robopoint_Base):
    """
    RoboPoint model implementation for GPU with quantization support.
    """

    def __init__(
        self,
        model_name: str = "wentao-yuan/robopoint-v1-vicuna-v1.5-13b",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_8bit: bool = True,
        **kwargs
    ):
        """
        Initialize the RoboPoint GPU implementation.
        
        Args:
            model_name: The name of the model to load.
            device: The device to use for inference.
            use_8bit: Whether to use 8-bit quantization.
        """
        self.model_name = model_name
        self.device = device
        self.use_8bit = use_8bit
        
        # Check if CUDA is available
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available, falling back to CPU")
            self.device = "cpu"
            
        print(f"Initializing RoboPoint GPU implementation with model: {model_name}")
        print(f"Device: {device}, Use 8-bit: {use_8bit}")
        print(f"Using Python {sys.version} environment at: {sys.prefix}")
        
        # Import PIL modules
        try:
            from PIL import Image, ImageDraw
            self.Image = Image
            self.ImageDraw = ImageDraw
        except ImportError:
            print("Warning: PIL modules could not be imported. Installing PIL...")
            import subprocess
            subprocess.run(["pip", "install", "pillow"], check=True)
            from PIL import Image, ImageDraw
            self.Image = Image
            self.ImageDraw = ImageDraw
        
        # Install required dependencies
        self._install_dependencies()
            
        # Load the model
        self._model, self._processor = self._load_model()

    def _install_dependencies(self):
        """
        Install required dependencies for the RoboPoint model.
        """
        try:
            print("Checking and installing required dependencies...")
            
            # Get the requirements file path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            requirements_path = os.path.join(current_dir, "requirements", "requirements_gpu.txt")
            
            if os.path.exists(requirements_path):
                with open(requirements_path, 'r') as f:
                    requirements = f.read().splitlines()
                
                # Check if transformers is installed
                try:
                    import transformers
                    print(f"Transformers version: {transformers.__version__}")
                except ImportError:
                    print("Installing transformers...")
                    import subprocess
                    subprocess.run([sys.executable, "-m", "pip", "install", "transformers"], check=True)
                
                # Check if torch is installed
                try:
                    import torch
                    print(f"PyTorch version: {torch.__version__}")
                except ImportError:
                    print("Installing PyTorch...")
                    import subprocess
                    subprocess.run([sys.executable, "-m", "pip", "install", "torch"], check=True)
                
                # Check if accelerate is installed
                try:
                    import accelerate
                    print(f"Accelerate version: {accelerate.__version__}")
                except ImportError:
                    print("Installing accelerate...")
                    import subprocess
                    subprocess.run([sys.executable, "-m", "pip", "install", "accelerate"], check=True)
                
                # Check if bitsandbytes is installed (for 8-bit quantization)
                if self.use_8bit:
                    try:
                        import bitsandbytes
                        print(f"BitsAndBytes version: {bitsandbytes.__version__}")
                    except ImportError:
                        print("Installing bitsandbytes...")
                        import subprocess
                        subprocess.run([sys.executable, "-m", "pip", "install", "bitsandbytes"], check=True)
                
                print("All dependencies installed successfully")
            else:
                print(f"Requirements file not found at {requirements_path}")
        except Exception as e:
            print(f"Error installing dependencies: {str(e)}")

    def _load_model(self):
        """
        Load the RoboPoint model from HuggingFace.
        """
        try:
            print(f"Loading RoboPoint model: {self.model_name}")
            
            # Import required libraries
            from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoProcessor
            
            # First try to load the processor
            try:
                processor = AutoProcessor.from_pretrained(self.model_name)
                print("Successfully loaded processor with AutoProcessor")
            except Exception as e:
                print(f"Error loading processor with AutoProcessor: {str(e)}")
                # Fallback to tokenizer
                processor = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
                print("Fallback: Successfully loaded tokenizer with AutoTokenizer")
            
            # Load the model with quantization if specified
            if self.use_8bit and self.device == "cuda":
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    load_in_8bit=True,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto" if self.device == "cuda" else None,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    trust_remote_code=True
                )
                
                if self.device == "cpu":
                    model = model.to(self.device)
            
            print(f"Successfully loaded RoboPoint model: {self.model_name}")
            return model, processor
            
        except Exception as e:
            print(f"Error loading RoboPoint model: {str(e)}")
            raise  # Re-raise the error without fallback

    def inference(self, image_path, text_instruction=None, output=None):
        """
        Run inference with the RoboPoint model to predict keypoint affordances.
        """
        print(f"Running inference on RoboPointGPU with model: {self.model_name}")
        print(f"Image: {image_path}")
        print(f"Instruction: {text_instruction}")
        
        try:
            # Load the image
            image = self.Image.open(image_path).convert("RGB")
            
            # Prepare the prompt
            if text_instruction is None:
                text_instruction = "In the image, identify and pinpoint several key locations. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image."
            
            # For LLaVA models, we need to use a specific format with image tokens
            prompt = f"<image>\n{text_instruction}"
            
            # Process image and text for the model
            try:
                # Try using the processor directly
                inputs = self._processor(
                    images=image,
                    text=prompt,
                    return_tensors="pt"
                ).to(self.device)
            except (AttributeError, TypeError) as e:
                print(f"Error with direct processor: {str(e)}")
                # Try alternative processing methods
                try:
                    # Method 1: Use separate image and text processing
                    from transformers import CLIPImageProcessor
                    
                    # Check if processor has image processing capability
                    if hasattr(self._processor, 'image_processor'):
                        image_processor = self._processor.image_processor
                    else:
                        # Try to get the image processor from the model config
                        vision_tower = getattr(self._model.config, 'vision_tower', 'openai/clip-vit-large-patch14')
                        image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
                    
                    # Process the image
                    image_inputs = image_processor(images=image, return_tensors="pt").to(self.device)
                    
                    # Process the text
                    if hasattr(self._processor, 'tokenizer'):
                        tokenizer = self._processor.tokenizer
                        text_inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
                    else:
                        text_inputs = self._processor(prompt, return_tensors="pt").to(self.device)
                    
                    # Combine inputs
                    inputs = {
                        **text_inputs,
                        "pixel_values": image_inputs.pixel_values
                    }
                except Exception as e2:
                    print(f"Error with alternative processing: {str(e2)}")
                    # Method 2: Simplest fallback - just use the text and let the model handle it
                    inputs = self._processor(prompt, return_tensors="pt").to(self.device)
            
            # Generate the output
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False
                )
            
            # Decode the output
            if hasattr(self._processor, 'decode'):
                response = self._processor.decode(output_ids[0], skip_special_tokens=True)
            elif hasattr(self._processor, 'batch_decode'):
                response = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            else:
                # Fallback to direct attribute access
                response = self._processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Extract the response part after the instruction
            response = response.split(text_instruction)[-1].strip()
            
            # Parse the keypoints from the response
            import re
            keypoints_str = re.findall(r'\[\(.*?\)\]', response)
            if keypoints_str:
                keypoints_str = keypoints_str[0]
                # Extract individual tuples
                tuples_str = re.findall(r'\(([^)]+)\)', keypoints_str)
                keypoints = []
                for t in tuples_str:
                    try:
                        x, y = map(float, t.split(','))
                        keypoints.append((x, y))
                    except:
                        continue
            else:
                # No keypoints found in the expected format
                raise ValueError(f"Could not parse keypoints from model response: {response}")
            
            # Visualize if output path is provided
            if output and keypoints:
                draw = self.ImageDraw.Draw(image)
                width, height = image.size
                
                for x, y in keypoints:
                    # Convert normalized coordinates to pixel coordinates
                    px = int(x * width)
                    py = int(y * height)
                    
                    # Draw a circle for each keypoint
                    radius = max(5, int(min(width, height) * 0.01))
                    draw.ellipse(
                        [(px - radius, py - radius), (px + radius, py + radius)],
                        fill=(255, 0, 0),
                        outline=(0, 0, 0)
                    )
                
                # Save the visualization
                os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
                image.save(output)
                print(f"Saved visualization to: {output}")
            
            return {
                "status": "success",
                "keypoints": keypoints,
                "raw_response": response
            }
            
        except Exception as e:
            print(f"Error in inference: {str(e)}")
            return {"status": "error", "message": str(e)} 