from ._base import Whisper_Base
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
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop(success=exc_type is None)
        return False

class Whisper_Jetson(Whisper_Base):
    def __init__(self, model_name="base.en"):
        """
        Initialize the Whisper model for Jetson using TensorRT.
        
        Args:
            model_name (str): The name of the Whisper model to use.
                Options include: 'tiny.en', 'base.en', 'small.en', 'medium.en', 'tiny', 'base', 'small', 'medium', 'large'
        """
        super().__init__()
        self.model_name = model_name
        self.model = None
        self.cache_dir = os.path.expanduser("~/.cache/whisper_trt")
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Check if we're running on a Jetson device
        if not self._is_jetson_device():
            print("Warning: Not running on a Jetson device. Falling back to GPU implementation.")
            from .whisper_gpu import Whisper_GPU
            self._gpu_impl = Whisper_GPU(model_name=model_name)
            return
            
        # Install dependencies if needed
        with ProgressIndicator("Installing dependencies...") as progress:
            self._install_dependencies()
            
        # Fix CUDA detection if needed
        self._try_fix_cuda_detection()
            
        # Load the model
        with ProgressIndicator(f"Loading Whisper TensorRT model '{model_name}'...") as progress:
            self._load_model()
            
    def _install_dependencies(self, verbose=False, force_nvidia_wheel=False):
        """Install required dependencies for Whisper TensorRT."""
        requirements_file = Path(__file__).parent / "requirements_jetson.txt"
        
        if not requirements_file.exists():
            # Create requirements file if it doesn't exist
            with open(requirements_file, "w") as f:
                f.write("numpy>=1.20.0\n")
                f.write("torch>=2.0.0\n")
                f.write("tqdm>=4.65.0\n")
                f.write("soundfile>=0.12.1\n")
                f.write("tensorrt>=8.5.0\n")
                
        # Install dependencies
        try:
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
            if verbose:
                subprocess.check_call(cmd)
            else:
                # Capture stderr to provide better error messages
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode != 0:
                    print(f"Warning: Failed to install some dependencies. Error: {result.stderr}")
                    print("Attempting to continue with available packages...")
                
            # Install whisper_trt
            try:
                cmd = [sys.executable, "-m", "pip", "install", "git+https://github.com/NVIDIA-AI-IOT/whisper_trt.git"]
                if verbose:
                    subprocess.check_call(cmd)
                else:
                    # Capture stderr to provide better error messages
                    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if result.returncode != 0:
                        print(f"Warning: Failed to install whisper_trt. Error: {result.stderr}")
                        print("Will fall back to standard whisper if available.")
            except Exception as e:
                print(f"Warning: Failed to install whisper_trt: {e}")
                print("Will fall back to standard whisper if available.")
                
            # Check if we need to install NVIDIA PyTorch wheels
            if force_nvidia_wheel or self._check_nvidia_pytorch():
                try:
                    self.install_nvidia_pytorch(verbose=verbose)
                except Exception as e:
                    print(f"Warning: Failed to install NVIDIA PyTorch wheels: {e}")
                    print("Will attempt to use standard PyTorch if available.")
                
        except subprocess.CalledProcessError as e:
            print(f"Warning: Error installing dependencies: {e}")
            print("Attempting to continue with available packages...")
        except Exception as e:
            print(f"Warning: Unexpected error during dependency installation: {e}")
            print("Attempting to continue with available packages...")
            
    @staticmethod
    def _is_jetson_device():
        """Check if we're running on a Jetson device."""
        try:
            # Check for Jetson-specific file
            return os.path.exists('/etc/nv_tegra_release')
        except:
            return False
            
    def _try_fix_cuda_detection(self, verbose=False):
        """Try to fix CUDA detection issues on Jetson."""
        try:
            # Check if CUDA_HOME is set
            if 'CUDA_HOME' not in os.environ:
                # Try to find CUDA path
                cuda_paths = [
                    '/usr/local/cuda',
                    '/usr/local/cuda-11.4',
                    '/usr/local/cuda-11.2',
                    '/usr/local/cuda-11.0',
                    '/usr/local/cuda-10.2',
                ]
                
                for path in cuda_paths:
                    if os.path.exists(path):
                        os.environ['CUDA_HOME'] = path
                        if verbose:
                            print(f"Set CUDA_HOME to {path}")
                        break
        except Exception as e:
            if verbose:
                print(f"Error fixing CUDA detection: {e}")
                
    def _load_model(self):
        """Load the Whisper TensorRT model."""
        try:
            # If we're using the GPU implementation as a fallback
            if hasattr(self, '_gpu_impl'):
                return
                
            # Try to import whisper_trt
            try:
                from whisper_trt import load_trt_model
                
                # Map model names to whisper_trt format if needed
                model_name = self.model_name
                
                # Load the model with TensorRT acceleration
                self.model = load_trt_model(model_name, path=os.path.join(self.cache_dir, f"{model_name}_trt.pth"))
                self._using_whisper_trt = True
                
            except ImportError:
                print("Warning: Failed to import whisper_trt. Falling back to standard whisper.")
                try:
                    import whisper
                    import torch
                    
                    self.model = whisper.load_model(self.model_name).to("cuda")
                    self._using_standard_whisper = True
                except Exception as e:
                    print(f"Warning: Failed to load standard Whisper model: {e}")
                    print("Using dummy transcription model as fallback.")
                    self.model = None
            except Exception as e:
                print(f"Warning: Error loading whisper_trt model: {e}")
                print("Falling back to standard whisper.")
                try:
                    import whisper
                    import torch
                    
                    self.model = whisper.load_model(self.model_name).to("cuda")
                    self._using_standard_whisper = True
                except Exception as e2:
                    print(f"Warning: Failed to load standard Whisper model: {e2}")
                    print("Using dummy transcription model as fallback.")
                    self.model = None
                    
        except Exception as e:
            print(f"Warning: Error loading Whisper model: {e}")
            print("Using dummy transcription model as fallback.")
            self.model = None
            
    def transcribe(self, audio_path, language=None, task="transcribe"):
        """
        Transcribe audio file to text.
        
        Args:
            audio_path (str): Path to the audio file or a numpy array of audio samples.
            language (str, optional): Language of the audio. If None, language is auto-detected.
            task (str, optional): Task to perform. Either 'transcribe' or 'translate'. Defaults to 'transcribe'.
            
        Returns:
            dict: Transcription result containing text and other metadata.
        """
        # If we're using the GPU implementation as a fallback
        if hasattr(self, '_gpu_impl'):
            return self._gpu_impl.transcribe(audio_path, language=language, task=task)
            
        # If model failed to load, return dummy result
        if self.model is None:
            print("Warning: Using dummy transcription result because model failed to load.")
            return {
                "text": "[Transcription unavailable - model failed to load]",
                "segments": [],
                "language": "en" if language is None else language
            }
            
        # Monitor resource usage during inference
        monitor = ResourceMonitor().start()
        
        try:
            with ProgressIndicator(f"Transcribing audio...") as progress:
                # Handle numpy array input
                if not isinstance(audio_path, str):
                    # For numpy array, we need to save it temporarily
                    import tempfile
                    import numpy as np
                    try:
                        import soundfile as sf
                        
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                            temp_path = temp_file.name
                            # Assuming the numpy array is in the format expected by soundfile
                            sf.write(temp_path, audio_path, 16000)
                            audio_data = temp_path
                    except ImportError:
                        print("Warning: soundfile not available. Cannot process numpy array input.")
                        return {
                            "text": "[Transcription error: soundfile not available for numpy array processing]",
                            "segments": [],
                            "language": "en" if language is None else language
                        }
                else:
                    # Use file path directly
                    audio_data = audio_path
                
                # Check which implementation we're using
                if hasattr(self, '_using_whisper_trt') and self._using_whisper_trt:
                    # whisper_trt implementation
                    result = self.model.transcribe(
                        audio_data,
                        language=language,
                        task=task
                    )
                else:
                    # Standard whisper implementation
                    import whisper
                    result = self.model.transcribe(
                        audio_data,
                        language=language,
                        task=task
                    )
                
                # Clean up temp file if created
                if not isinstance(audio_path, str) and 'temp_path' in locals():
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                
                # Add resource usage information
                result["resource_usage"] = monitor.get_stats()
                
                return result
                
        except Exception as e:
            print(f"Error during transcription: {e}")
            return {
                "text": f"[Transcription error: {str(e)}]",
                "segments": [],
                "language": "en" if language is None else language,
                "error": str(e)
            }
        finally:
            monitor.stop()
            
    def _check_nvidia_pytorch(self):
        """Check if we need to install NVIDIA PyTorch wheels."""
        try:
            import torch
            return False  # Already installed
        except ImportError:
            return True  # Need to install
            
    def install_nvidia_pytorch(self, verbose=False):
        """Install NVIDIA PyTorch wheels for Jetson."""
        try:
            # Install PyTorch for Jetson
            cmd = [
                sys.executable, "-m", "pip", "install", 
                "--no-cache-dir",
                "torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1",
                "-f", "https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl"
            ]
            
            if verbose:
                subprocess.check_call(cmd)
            else:
                subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
        except subprocess.CalledProcessError as e:
            print(f"Error installing NVIDIA PyTorch: {e}")
            raise 