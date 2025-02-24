from ._base import Clip_Base
import os
import subprocess
from PIL import Image
import time
import sys
import threading
import itertools
import json
import tempfile
from pathlib import Path

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

class Clip_Jetson(Clip_Base):
    def __init__(self, model_name="openai/clip-vit-large-patch14-336"):
        """
        Initializes CLIP model on Jetson using the clip_trt package directly.
        
        Args:
            model_name (str): Name of the CLIP model to use (from HuggingFace)
        """
        self.model_name = model_name
        
        # Create cache directory for model downloads
        self.cache_dir = Path.home() / ".cache" / "exla" / "clip_trt"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Install dependencies
        self._install_dependencies()
        
        # Initialize model
        self.model = None

    def _install_dependencies(self):
        """Install required dependencies for clip_trt"""
        with ProgressIndicator("Installing clip_trt dependencies") as progress:
            try:
                # First check if clip_trt is already installed
                try:
                    import clip_trt
                    progress.stop(final_message="clip_trt already installed")
                    return
                except ImportError:
                    pass
                
                # Install required dependencies first
                subprocess.run([
                   "uv", "pip", "install", "pillow", "torch", "transformers", "psutil"
                ], check=True)
                
                # Install torch2trt first (required by clip_trt)
                subprocess.run([
                    "uv", "pip", "install", "git+https://github.com/NVIDIA-AI-IOT/torch2trt.git"
                ], check=True)
                
                # Set environment variables to ensure dependencies are available during installation
                env = os.environ.copy()
                env["PYTHONPATH"] = os.path.dirname(os.path.dirname(sys.executable)) + ":" + env.get("PYTHONPATH", "")
                
                # Install from GitHub if not installed
                subprocess.run([
                    "uv", "pip", "install", "--no-deps", "git+https://github.com/dusty-nv/clip_trt.git"
                ], env=env, check=True)
                
                progress.stop(final_message="Successfully installed clip_trt")
            except Exception as e:
                progress.stop(success=False, final_message=f"Failed to install dependencies: {e}")
                raise

    def _load_model(self):
        """Load the CLIP model using clip_trt"""
        if self.model is not None:
            return
            
        with ProgressIndicator(f"Loading CLIP model: {self.model_name}") as progress:
            try:
                from clip_trt import CLIPModel
                
                # Load the model with TensorRT optimization
                self.model = CLIPModel.from_pretrained(
                    self.model_name,
                    use_tensorrt=True,
                    crop=False,
                )
                progress.stop(final_message="Model loaded successfully")
            except Exception as e:
                progress.stop(success=False, final_message=f"Failed to load model: {e}")
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

    def inference(self, image_paths, text_queries=[], timeout=300, debug=False):
        """
        Runs CLIP inference using the clip_trt package directly.
        
        Args:
            image_paths: String or list of image paths
            text_queries: List of text queries to compare against
            timeout: Maximum time in seconds to wait for inference
            debug: Whether to print detailed debug information
            
        Returns:
            List of dictionaries containing predictions for each text query
        """
        print("\n🚀 Starting Exla CLIP Inference Pipeline\n")
        
        # Track overall execution time
        total_start_time = time.time()
        timings = {}
        
        # Process images
        with ProgressIndicator("Loading and preprocessing images") as progress:
            valid_paths, pil_images = self._load_images(image_paths)
            if not valid_paths:
                progress.stop(success=False, final_message="No valid images found")
                return {"error": "No valid images found"}
            timings["image_processing"] = progress.stop(final_message=f"Processed {len(valid_paths)} images")
        
        try:
            # Load the model if not already loaded
            model_load_start = time.time()
            self._load_model()
            timings["model_loading"] = f"{time.time() - model_load_start:.1f}s"
            
            # Run inference
            with ProgressIndicator("Running CLIP inference") as progress:
                inference_start = time.time()
                
                # Get similarity scores
                similarity = self.model(pil_images, text_queries)
                
                if debug:
                    print(f"DEBUG: Similarity shape: {similarity.shape}")
                    print(f"DEBUG: Similarity values: {similarity}")
                
                timings["inference"] = progress.stop(final_message="Inference completed successfully")
            
            # Process results
            with ProgressIndicator("Processing results") as progress:
                results = []
                
                # Convert similarity matrix to the expected format
                # similarity is a 2D tensor with shape [num_texts, num_images]
                import torch
                similarity_np = similarity.cpu().numpy() if isinstance(similarity, torch.Tensor) else similarity
                
                for i, query in enumerate(text_queries):
                    matches = []
                    
                    # Get scores for this text query across all images
                    scores = similarity_np[i]
                    
                    # Match scores with image paths
                    for j, (img_path, score) in enumerate(zip(valid_paths, scores)):
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
            print(f"   • Images processed: {len(valid_paths)}")
            print(f"   • Text queries: {len(text_queries)}")
            print(f"   • Total time: {total_time:.2f}s")
            print("\n⏱️  Timing Breakdown:")
            for step, duration in timings.items():
                print(f"   • {step.replace('_', ' ').title()}: {duration}")
            
            return results
            
        except Exception as e:
            print(f"\n❌ Error running CLIP inference: {e}")
            import traceback
            traceback.print_exc()
            raise
    