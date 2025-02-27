from ._base import Whisper_Base
import os
import subprocess
import sys
import time
import json
import threading
import itertools
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

class Whisper_GPU(Whisper_Base):
    def __init__(self, model_name="base.en"):
        """
        Initialize the Whisper model for GPU.
        
        Args:
            model_name (str): The name of the Whisper model to use.
                Options include: 'tiny.en', 'base.en', 'small.en', 'medium.en', 'tiny', 'base', 'small', 'medium', 'large'
        """
        super().__init__()
        self.model_name = model_name
        self.model = None
        
        # Install dependencies if needed
        with ProgressIndicator("Installing dependencies...") as progress:
            self._install_dependencies()
            
        # Load the model
        with ProgressIndicator(f"Loading Whisper model '{model_name}'...") as progress:
            self._load_model()
            
    def _install_dependencies(self, verbose=False):
        """Install required dependencies for Whisper."""
        requirements_file = Path(__file__).parent / "requirements_gpu.txt"
        
        if not requirements_file.exists():
            # Create requirements file if it doesn't exist
            with open(requirements_file, "w") as f:
                f.write("openai-whisper>=20231117\n")
                f.write("numpy>=1.20.0\n")
                f.write("torch>=2.0.0\n")
                f.write("tqdm>=4.65.0\n")
                f.write("faster-whisper>=0.9.0\n")  # Optimized implementation
                f.write("soundfile>=0.12.1\n")
        
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
        except subprocess.CalledProcessError as e:
            print(f"Warning: Error installing dependencies: {e}")
            print("Attempting to continue with available packages...")
        except Exception as e:
            print(f"Warning: Unexpected error during dependency installation: {e}")
            print("Attempting to continue with available packages...")
            
    def _load_model(self):
        """Load the Whisper model."""
        try:
            # Use faster-whisper for GPU acceleration
            from faster_whisper import WhisperModel
            
            # Map model names to faster-whisper format if needed
            model_name = self.model_name
            
            # Load the model with GPU acceleration
            self.model = WhisperModel(model_name, device="cuda", compute_type="float16")
        except ImportError:
            print("Warning: Failed to import faster-whisper. Falling back to standard whisper.")
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
                # Check if we're using standard whisper or faster-whisper
                if hasattr(self, '_using_standard_whisper') and self._using_standard_whisper:
                    # Standard whisper implementation
                    import whisper
                    
                    # Handle numpy array input
                    if not isinstance(audio_path, str):
                        # Assume it's a numpy array
                        audio_data = audio_path
                    else:
                        # Load audio from file path
                        audio_data = audio_path
                    
                    # Run transcription
                    result = self.model.transcribe(
                        audio_data,
                        language=language,
                        task=task
                    )
                else:
                    # faster-whisper implementation
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
                    
                    # Run transcription with faster-whisper
                    segments, info = self.model.transcribe(
                        audio_data,
                        language=language,
                        task=task,
                        beam_size=5
                    )
                    
                    # Convert to format similar to whisper's output
                    result = {
                        "text": " ".join([segment.text for segment in segments]),
                        "segments": [{"text": segment.text, "start": segment.start, "end": segment.end} for segment in segments],
                        "language": info.language,
                        "language_probability": info.language_probability
                    }
                    
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