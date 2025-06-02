from ._base import Clip_Base
import os
import subprocess
import sys
import time
import json
import threading
import itertools
from pathlib import Path
from inferx.utils.resource_monitor import ResourceMonitor

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

class Clip_CPU(Clip_Base):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Initializes a CLIP model on CPU using the transformers library.
        """
        # Get device information
        self.resource_monitor = ResourceMonitor()
        device_type = self.resource_monitor.device_type
        
        # Print minimal initialization message
        print(f"\n✨ InferX - CLIP Model ✨")
        print(f"🔍 Device Detected: {device_type.upper()}")
        
        # Print initial resource usage
        print("\n📊 Initial System Resources:")
        self.resource_monitor.get_memory_usage(print_info=True)
        
        # Set model properties
        self.model_name = model_name
        
        # Print initialization message with animation
        with ProgressIndicator(f"Initializing InferX Optimized CLIP model for CPU [CPU Mode]") as progress:
            # Install dependencies (will only do the minimum necessary)
            self._install_dependencies(verbose=False)
            progress.stop()
        
        self.model = None
        self.processor = None

    def _install_dependencies(self, verbose=False):
        """
        Install dependencies required for CLIP on CPU.
        
        Args:
            verbose (bool): Whether to print verbose output
        """
        try:
            # Get the path to the requirements file
            current_dir = Path(__file__).parent
            requirements_file = current_dir / "requirements" / "requirements_cpu.txt"
            
            if not requirements_file.exists():
                if verbose:
                    print(f"Requirements file not found: {requirements_file}")
                return False
                
            # Check if required packages are installed
            try:
                from PIL import Image
                import transformers
                import torch
                import psutil
                
                if verbose:
                    print("All required packages are already installed")
                return True
            except ImportError:
                if verbose:
                    print(f"Installing dependencies from {requirements_file}")
                
                # Install requirements
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                    stdout=subprocess.DEVNULL if not verbose else None,
                    stderr=subprocess.DEVNULL if not verbose else None
                )
                
                if verbose:
                    print("✓ Dependencies installed successfully")
                return True
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to install dependencies: {str(e)}")
                print("The model will still work, but may have reduced functionality.")
            return False

    def _load_model(self):
        """Load the CLIP model using transformers"""
        if self.model is not None:
            return
            
        with ProgressIndicator("Loading CLIP model") as progress:
            try:
                from transformers import CLIPProcessor, CLIPModel
                
                # Load the model
                self.model = CLIPModel.from_pretrained(self.model_name)
                self.processor = CLIPProcessor.from_pretrained(self.model_name)
                
                # Print memory usage after model loading
                print("\nMemory usage after model loading:")
                self.resource_monitor.get_memory_usage(print_info=True)
                
                progress.stop(final_message="Model ready on CPU")
            except Exception as e:
                progress.stop(success=False, final_message=f"Failed to load model: {e}")
                raise

    def _load_images(self, image_input):
        """
        Loads images from paths and returns valid paths and PIL images.
        """
        # Import PIL here to avoid import at module level
        from PIL import Image
        
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
        timings = {}
        
        # Process images
        with ProgressIndicator("Processing images") as progress:
            valid_paths, pil_images = self._load_images(image_paths)
            if not valid_paths:
                progress.stop(success=False, final_message="No valid images found")
                return None
            timings["image_processing"] = progress.stop(final_message=f"Processed {len(valid_paths)} images")
        
        try:
            # Load the model if not already loaded
            model_load_start = time.time()
            self._load_model()
            timings["model_loading"] = f"{time.time() - model_load_start:.1f}s"
            
            # Run inference
            with ProgressIndicator("Running CLIP inference") as progress:
                inference_start = time.time()
                
                # Process inputs
                inputs = self.processor(
                    text=text_queries,
                    images=pil_images,
                    return_tensors="pt",
                    padding=True
                )
                
                # Print memory usage before inference
                print("\n📊 Memory usage before inference:")
                self.resource_monitor.get_memory_usage(print_info=True)
                
                # Get similarity scores
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image.detach().numpy()
                
                # Print memory usage after inference
                print("\n📊 Memory usage after inference:")
                self.resource_monitor.get_memory_usage(print_info=True)
                
                timings["inference"] = progress.stop(final_message="Inference completed successfully")
            
            # Process results
            with ProgressIndicator("Processing results") as progress:
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
                
                timings["processing"] = progress.stop()
            
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
            for step, duration in timings.items():
                print(f"   • {step.replace('_', ' ').title()}: {duration}")
            
            # Print final resource usage
            print("\n📊 Final System Resources:")
            self.resource_monitor.get_memory_usage(print_info=True)
            
            return results
            
        except Exception as e:
            print(f"\n❌ Error running CLIP inference: {e}")
            import traceback
            traceback.print_exc()
            return None
       