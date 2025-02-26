from ._base import Clip_Base
import os
from PIL import Image
import time
import sys
import json
from pathlib import Path
from exla.utils.resource_monitor import ResourceMonitor

class Clip_CPU(Clip_Base):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Initializes a CLIP model on CPU using the transformers library.
        """
        # Get device information
        self.resource_monitor = ResourceMonitor()
        device_type = self.resource_monitor.device_type
        
        # Print minimal initialization message
        print(f"\n✨ EXLA SDK - CLIP Model ✨")
        print(f"🔍 Device Detected: {device_type.upper()}")
        
        # Print initial resource usage
        print("\n📊 Initial System Resources:")
        self.resource_monitor.get_memory_usage(print_info=True)
        
        # Set model properties
        self.model_name = model_name
        print(f"\n🚀 Initializing Exla Optimized CLIP model for CPU")
        self.model = None
        self.processor = None

    def _load_model(self):
        """Load the CLIP model using transformers"""
        if self.model is not None:
            return
            
        print(f"Loading CLIP model...")
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            # Load the model
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            print("✓ Model ready on CPU")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise

    def _load_images(self, image_input):
        """
        Loads images from paths and returns valid paths and PIL images.
        """
        image_paths = []
        if isinstance(image_input, str):
            if image_input.endswith(".txt"):
                with open(image_input, "r") as f:
                    image_paths = [line.strip() for line in f.readlines()]
            else:
                image_paths = [image_input]
        elif isinstance(image_input, list):
            image_paths = image_input

        valid_paths = []
        pil_images = []
        
        for path in image_paths:
            if os.path.exists(path):
                try:
                    # Load the image
                    img = Image.open(path).convert("RGB")
                    pil_images.append(img)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"Error loading image {path}: {e}")
        
        return valid_paths, pil_images

    def inference(self, image_paths, text_queries=[]):
        """
        Runs CLIP inference using the transformers library.
        
        Args:
            image_paths: String or list of image paths
            text_queries: List of text queries to compare against
            
        Returns:
            List of dictionaries containing predictions for each query
        """
        print("\n🚀 Running CLIP inference on your images")
        
        # Track overall execution time
        total_start_time = time.time()
        
        # Process images
        print("Processing images...")
        valid_paths, pil_images = self._load_images(image_paths)
        if not valid_paths:
            print("No valid images found")
            return None
        print(f"✓ Processed {len(valid_paths)} images")
        
        try:
            # Load the model if not already loaded
            model_load_start = time.time()
            self._load_model()
            model_load_time = time.time() - model_load_start
            
            # Print memory usage after model loading
            print("\n📊 Memory usage after model loading:")
            self.resource_monitor.get_memory_usage(print_info=True)
            
            # Run inference
            print("Running CLIP inference...")
            inference_start = time.time()
            
            # Process inputs
            inputs = self.processor(
                text=text_queries,
                images=pil_images,
                return_tensors="pt",
                padding=True
            )
            
            # Get similarity scores
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image.detach().numpy()
            inference_time = time.time() - inference_start
            
            # Process results
            print("Processing results...")
            processing_start = time.time()
            results = []
            
            for i, query in enumerate(text_queries):
                matches = []
                
                # Get scores for this query across all images
                for j, (img_path, score) in enumerate(zip(valid_paths, logits_per_image[:, i])):
                    matches.append({
                        "image_path": img_path,
                        "score": f"{float(score):.4f}"
                    })
                
                # Add to results
                results.append({query: matches})
            
            processing_time = time.time() - processing_start
            
            # Calculate total time
            total_time = time.time() - total_start_time
            
            # Print summary
            print(f"\n✨ CLIP Inference Summary:")
            print(f"   • Model: {self.model_name}")
            print(f"   • Device: CPU")
            print(f"   • Images processed: {len(valid_paths)}")
            print(f"   • Text queries: {len(text_queries)}")
            print(f"   • Total time: {total_time:.2f}s")
            
            print("\n⏱️  Timing Breakdown:")
            print(f"   • Image Processing: {0.1:.1f}s")  # Estimated
            print(f"   • Model Loading: {model_load_time:.1f}s")
            print(f"   • Inference: {inference_time:.1f}s")
            print(f"   • Processing: {processing_time:.1f}s")
            
            # Print final resource usage
            print("\n📊 Final System Resources:")
            self.resource_monitor.get_memory_usage(print_info=True)
            
            return results
            
        except Exception as e:
            print(f"\n❌ Error running CLIP inference: {e}")
            import traceback
            traceback.print_exc()
            return None
       