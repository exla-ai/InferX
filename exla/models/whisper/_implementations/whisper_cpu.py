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

class Whisper_CPU(Whisper_Base):
    def __init__(self, model_name="base.en"):
        """
        Initialize the Whisper model for CPU.
        
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
        requirements_file = Path(__file__).parent / "requirements_cpu.txt"
        
        if not requirements_file.exists():
            # Create requirements file if it doesn't exist
            with open(requirements_file, "w") as f:
                f.write("openai-whisper>=20231117\n")
                f.write("numpy>=1.20.0\n")
                f.write("torch>=2.0.0\n")
                f.write("tqdm>=4.65.0\n")
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
            import whisper
            self.model = whisper.load_model(self.model_name)
        except ImportError:
            print("Error: Failed to import whisper. Please make sure it's installed correctly.")
            print("Using dummy transcription model as fallback.")
            self.model = None
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
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