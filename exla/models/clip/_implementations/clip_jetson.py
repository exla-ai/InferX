from ._base import Clip_Base
import os
import subprocess
import sys
import threading
import itertools
import json
import tempfile
import time
from pathlib import Path
from exla.utils.resource_monitor import ResourceMonitor

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
        Initializes CLIP model on Jetson using transformers with GPU acceleration if available.
        
        Args:
            model_name (str): Name of the CLIP model to use (from HuggingFace)
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
        
        # Create cache directory for model downloads
        self.cache_dir = Path.home() / ".cache" / "exla" / "clip_trt"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Print initialization message with animation
        with ProgressIndicator(f"Initializing Exla Optimized CLIP model for {device_type.upper()} [GPU Mode]") as progress:
            # Install dependencies (will only do the minimum necessary)
            self._install_dependencies(verbose=False)
            progress.stop()
        
        # Initialize model
        self.model = None
        self.device = "cpu"  # Default to CPU, will be updated during model loading

    def _install_dependencies(self, verbose=False, force_nvidia_wheel=False):
        """
        Install dependencies required for CLIP on Jetson devices.
        
        Args:
            verbose (bool): Whether to print verbose output
            force_nvidia_wheel (bool): Whether to force installation of NVIDIA's PyTorch wheel
        """
        try:
            # Set CUDA environment variables if CUDA directory exists
            cuda_home = "/usr/local/cuda"
            if os.path.exists(cuda_home):
                os.environ["CUDA_HOME"] = cuda_home
                
                # Update LD_LIBRARY_PATH
                ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
                cuda_lib_path = f"{cuda_home}/lib64"
                if cuda_lib_path not in ld_library_path:
                    os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib_path}:{ld_library_path}"
                
                if verbose:
                    print(f"Set CUDA environment variables:")
                    print(f"  CUDA_HOME = {os.environ.get('CUDA_HOME')}")
                    print(f"  LD_LIBRARY_PATH = {os.environ.get('LD_LIBRARY_PATH')}")

            # Check Python version
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            is_python_310 = python_version == "3.10"
            
            if not is_python_310:
                print(f"⚠️ Python {python_version} is not optimal for Jetson GPU acceleration")
                print("For best performance, use Python 3.10 with NVIDIA's PyTorch wheel")
                print("See README for setup instructions")
            
            # Get the path to the requirements file
            current_dir = Path(__file__).parent
            requirements_file = current_dir / "requirements" / "requirements_jetson.txt"
            
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
                
                # If we need to force NVIDIA wheel installation, raise ImportError
                if force_nvidia_wheel:
                    raise ImportError("Forcing NVIDIA PyTorch wheel installation")
                
                # Check if PyTorch with CUDA is available
                if not torch.cuda.is_available() and self._is_jetson_device() and is_python_310:
                    if verbose:
                        print("CUDA is not available on this Jetson device.")
                    
                    # Install NVIDIA PyTorch wheel for Python 3.10
                    if verbose:
                        print("Installing NVIDIA's PyTorch wheel for optimal performance...")
                    
                    # Install from requirements file
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                        stdout=subprocess.DEVNULL if not verbose else None,
                        stderr=subprocess.DEVNULL if not verbose else None
                    )
                    
                    # Reload torch to use the new installation
                    import importlib
                    importlib.reload(torch)
                    
                    if torch.cuda.is_available():
                        print(f"✓ CUDA is now available with device: {torch.cuda.get_device_name(0)}")
                    else:
                        print("⚠️ CUDA is still not available after installing NVIDIA's PyTorch wheel")
                        print("Please check your CUDA installation and environment variables")
                else:
                    if verbose:
                        print("All required packages are already installed")
                    return True
            except ImportError:
                if verbose:
                    print(f"Installing dependencies from {requirements_file}")
                
                # Install from requirements file
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                    stdout=subprocess.DEVNULL if not verbose else None,
                    stderr=subprocess.DEVNULL if not verbose else None
                )
                
                if verbose:
                    print("✓ Dependencies installed successfully")
                
                # Check if CUDA is available after installation
                try:
                    import torch
                    if torch.cuda.is_available():
                        print(f"✓ CUDA is now available with device: {torch.cuda.get_device_name(0)}")
                    elif self._is_jetson_device() and is_python_310:
                        print("⚠️ CUDA is still not available after installing NVIDIA's PyTorch wheel")
                        print("Please check your CUDA installation and environment variables")
                except ImportError:
                    if verbose:
                        print("⚠️ Failed to import PyTorch after installation")
            
            return True
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to install dependencies: {str(e)}")
                print("The model will still work, but may have reduced functionality.")
            return False

    @staticmethod
    def _is_jetson_device():
        """Check if the current device is a Jetson."""
        try:
            # Check for Jetson-specific hardware
            if os.path.exists("/proc/device-tree/model"):
                with open("/proc/device-tree/model") as f:
                    model = f.read()
                    if "NVIDIA Jetson" in model:
                        return True
            
            # Check using nvidia-smi
            try:
                output = subprocess.check_output(["nvidia-smi", "-L"], universal_newlines=True)
                if "Jetson" in output or "Tegra" in output or "AGX" in output or "Orin" in output:
                    print("NVIDIA GPU detected via nvidia-smi")
                    return True
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
                
            return False
        except Exception:
            return False

    def _try_fix_cuda_detection(self, verbose=False):
        """Attempt to fix CUDA detection issues on Jetson devices."""
        try:
            # Check Python version
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            if verbose:
                print(f"Python version: {python_version}")
            
            if sys.version_info.major == 3 and sys.version_info.minor > 10:
                if verbose:
                    print(f"Current Python version {sys.version_info.major}.{sys.version_info.minor} may not be compatible with NVIDIA's PyTorch wheel")
                    print("For optimal performance, consider using Python 3.10 with NVIDIA's PyTorch wheel")
            
            # We could attempt to install the NVIDIA PyTorch wheel here,
            # but it's better to let the user decide since it's a significant change
            return False
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to fix CUDA detection: {str(e)}")
            return False

    def _load_model(self):
        """Load the CLIP model using transformers with CUDA if available"""
        if self.model is not None:
            return
            
        with ProgressIndicator(f"Loading CLIP model") as progress:
            try:
                # Import required libraries
                try:
                    from transformers import CLIPProcessor, CLIPModel
                    import torch
                except ImportError:
                    progress.stop(success=False, final_message="Required libraries not found")
                    print("Installing missing dependencies...")
                    subprocess.run([
                        "pip", "install", "transformers", "torch"
                    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    from transformers import CLIPProcessor, CLIPModel
                    import torch
                
                # Force CUDA detection for Jetson devices
                if self._is_jetson_device():
                    # Set environment variables that might help with CUDA detection
                    if os.path.exists("/usr/local/cuda"):
                        os.environ["CUDA_HOME"] = "/usr/local/cuda"
                        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                        
                        # Try to force PyTorch to recognize CUDA
                        try:
                            # This is a workaround to force PyTorch to re-check CUDA
                            torch.cuda.init()
                        except:
                            pass
                
                # Check CUDA availability again
                cuda_available = torch.cuda.is_available()
                if cuda_available:
                    device_name = torch.cuda.get_device_name(0)
                    print(f"Using GPU: {device_name}")
                    device = "cuda"
                    
                    # Print GPU memory information
                    self.resource_monitor.get_memory_usage(print_info=True)
                else:
                    print("Using CPU for inference")
                    device = "cpu"
                
                # Load the model with error handling
                try:
                    self.model = CLIPModel.from_pretrained(self.model_name)
                    self.processor = CLIPProcessor.from_pretrained(self.model_name)
                except Exception as e:
                    progress.stop(success=False, final_message=f"Error loading model: {e}")
                    print("Trying to load model with additional options...")
                    self.model = CLIPModel.from_pretrained(self.model_name, local_files_only=False, force_download=True)
                    self.processor = CLIPProcessor.from_pretrained(self.model_name, local_files_only=False, force_download=True)
                
                # Move model to appropriate device
                try:
                    self.model = self.model.to(device)
                    self.device = device
                except Exception as e:
                    print(f"Error moving model to {device}: {e}")
                    print("Falling back to CPU")
                    self.model = self.model.to("cpu")
                    self.device = "cpu"
                
                # Print memory usage after model loading
                if cuda_available:
                    print("\nMemory usage after model loading:")
                    self.resource_monitor.get_memory_usage(print_info=True)
                
                progress.stop(final_message=f"Model ready on {self.device.upper()}")
            except Exception as e:
                progress.stop(success=False, final_message=f"Failed to load model: {e}")
                print("Please check your internet connection and try again.")
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

    def inference(self, image_paths, text_queries=[], timeout=300, debug=False):
        """
        Runs CLIP inference using transformers with GPU acceleration if available.
        
        Args:
            image_paths: String or list of image paths
            text_queries: List of text queries to compare against
            timeout: Maximum time in seconds to wait for inference
            debug: Whether to print detailed debug information
            
        Returns:
            List of dictionaries containing predictions for each text query
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
                
                # Process inputs
                import torch
                inputs = self.processor(
                    text=text_queries,
                    images=pil_images,
                    return_tensors="pt",
                    padding=True
                )
                
                # Move inputs to the same device as the model
                device = getattr(self, 'device', 'cpu')
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Print memory usage before inference
                if device == "cuda":
                    print("\n📊 Memory usage before inference:")
                    self.resource_monitor.get_memory_usage(print_info=True)
                
                # Get similarity scores
                with torch.no_grad():  # Disable gradient calculation for inference
                    outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                
                # Move back to CPU for processing
                similarity = logits_per_image.detach().cpu().numpy()
                
                # Print memory usage after inference
                if device == "cuda":
                    print("\n📊 Memory usage after inference:")
                    self.resource_monitor.get_memory_usage(print_info=True)
                
                if debug:
                    print(f"DEBUG: Similarity shape: {similarity.shape}")
                    print(f"DEBUG: Similarity values: {similarity}")
                
                timings["inference"] = progress.stop(final_message="Inference completed successfully")
            
            # Process results
            with ProgressIndicator("Processing results") as progress:
                results = []
                
                for i, query in enumerate(text_queries):
                    matches = []
                    
                    # Get scores for this text query across all images
                    scores = similarity[i] if similarity.shape[0] == len(text_queries) else similarity[:, i]
                    
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
            print(f"   • Device: {getattr(self, 'device', 'cpu').upper()}")
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
            raise

    def _check_nvidia_pytorch(self):
        """
        Check if the NVIDIA PyTorch wheel is installed and provide instructions if not.
        Returns True if the correct PyTorch version is installed, False otherwise.
        """
        try:
            import torch
            if "nv" in torch.__version__:
                print(f"✓ Using NVIDIA PyTorch build: {torch.__version__}")
                return True
            else:
                print(f"⚠️ Not using NVIDIA's PyTorch build for Jetson (current: {torch.__version__})")
                print("For best GPU performance, install the NVIDIA PyTorch wheel:")
                print("pip install https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl")
                return False
        except ImportError:
            print("❌ PyTorch not installed")
            print("Please install the NVIDIA PyTorch wheel:")
            print("pip install https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl")
            return False

    def install_nvidia_pytorch(self):
        """
        Explicitly install NVIDIA's PyTorch wheel for optimal performance on Jetson devices.
        This method should be called if you want to force the installation of the NVIDIA PyTorch wheel.
        
        Returns:
            bool: True if installation was successful, False otherwise
        """
        print("\n🚀 Installing NVIDIA PyTorch for Jetson...")
        return self._install_dependencies(verbose=True, force_nvidia_wheel=True)
    