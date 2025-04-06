import os
import subprocess
import time
from pathlib import Path
import tempfile
import json
from typing import List, Dict, Union, Any, Optional

class InternVL2_5_MPS:
    """
    Implementation of InternVL2.5 model for Apple Silicon (MPS) devices.
    InternVL2.5 is a multimodal large language model that can process both
    images and text inputs for various tasks including visual reasoning,
    image captioning, visual question answering, and more.
    """
    
    def __init__(self, model_size: str = "1B", use_mpo: bool = False):
        """
        Initialize the InternVL2.5 model for MPS (Apple Silicon).
        
        Args:
            model_size: Size variant of the model to use ("1B", "2B", "4B", "8B").
                        Larger models (26B+) are not supported on MPS.
            use_mpo: Whether to use the MPO (Mixed Preference Optimization) version
                    of the model, which has better reasoning capabilities.
        """
        self.model_size = model_size
        self.use_mpo = use_mpo
        self.model = None
        self.processor = None
        self.tokenizer = None
        
        # Validate model size for MPS
        valid_sizes = ["1B", "2B", "4B", "8B"]
        if model_size not in valid_sizes:
            raise ValueError(f"Model size {model_size} not supported for MPS. "
                           f"Choose from {valid_sizes}")
        
        # self._install_dependencies()
        self._load_model()
    
    def _install_dependencies(self):
        """
        Install required dependencies for InternVL2.5.
        """
        try:
            subprocess.run([
                "pip", "install", "torch", "transformers>=4.36.0", "pillow", "accelerate"
            ], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to install dependencies: {str(e)}")
    
    def _load_model(self):
        """
        Load the InternVL2.5 model and processor.
        """
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModel
            import torch
            
            # Determine model name based on size and MPO option
            model_name = f"OpenGVLab/InternVL2_5-{self.model_size}"
            if self.use_mpo:
                model_name += "-MPO"
            
            print(f"Loading {model_name}...")
            
            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=False
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="mps",
                trust_remote_code=True
            )
            
            print(f"Model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load InternVL2.5 model: {str(e)}")
    
    def inference(self, image_paths: List[str], text_queries: List[str] = []) -> List[Dict]:
        """
        Run inference with the InternVL2.5 model.
        
        Args:
            image_paths: List of paths to images to process
            text_queries: List of text prompts/queries to process with the images
                         If empty, will generate captions for the images
        
        Returns:
            List of dictionaries containing model outputs for each query or image
        """
        if not self.model or not self.processor or not self.tokenizer:
            raise RuntimeError("Model not loaded. Initialize the model first.")
        
        if not image_paths:
            raise ValueError("At least one image path must be provided")
        
        try:
            from PIL import Image
            import torch
            import torchvision.transforms as T
            from torchvision.transforms.functional import InterpolationMode
            
            # Load images
            images = [Image.open(img_path).convert('RGB') for img_path in image_paths]
            
            # If no text queries provided, use a default caption generation prompt
            if not text_queries:
                text_queries = ["Describe this image in detail."]
            
            # Define image transformation function
            def build_transform(input_size=448):
                IMAGENET_MEAN = (0.485, 0.456, 0.406)
                IMAGENET_STD = (0.229, 0.224, 0.225)
                transform = T.Compose([
                    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                    T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                ])
                return transform
            
            # Process images with the correct transformation
            transform = build_transform(input_size=448)
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values).to(torch.float16).to("mps")
            
            results = []
            
            for query in text_queries:
                try:
                    # Format the query with image tag as shown in the documentation
                    prompt = f"<image>\n{query}"
                    
                    # Generate response using the chat method from the model
                    generation_config = dict(
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )
                    
                    with torch.no_grad():
                        response = self.model.chat(
                            self.tokenizer,
                            pixel_values,
                            prompt,
                            generation_config
                        )
                    
                except Exception as e:
                    print(f"Error processing query '{query}': {str(e)}")
                    response = f"Error: {str(e)}"
                    
                    # Try alternative approach if the first one fails
                    try:
                        # Try using the processor directly
                        inputs = self.processor(text=prompt, images=images, return_tensors="pt").to("mps")
                        
                        with torch.no_grad():
                            output = self.model.generate(
                                **inputs,
                                max_new_tokens=512,
                                do_sample=True,
                                temperature=0.7,
                                top_p=0.9,
                            )
                        
                        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    except Exception as nested_e:
                        print(f"Alternative approach also failed: {str(nested_e)}")
                
                results.append({
                    "query": query,
                    "response": response,
                    "images": image_paths
                })
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Inference failed: {str(e)}")

    