"""
RoboPoint model implementation for GPU.
"""

import os
import sys
import subprocess
import time
import threading
import itertools
import logging
from typing import Dict, Any
import requests
import io
import base64
import re
import json
from PIL import Image, ImageDraw

from ._base import Robopoint_Base
from .helpers import conv_templates
from .helpers import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


# Default Docker Hub repository for the RoboPoint image
DEFAULT_DOCKER_REPO = "public.ecr.aws/h1f5g0k2/exla:robopoint-v2-gpu-latest"

# Create logger but don't configure it yet
logger = logging.getLogger(__name__)

# Configure logging with minimal format (message only)
logging.basicConfig(
    level=logging.WARNING,  # Default level
    format='%(message)s'    # Only show the message, no prefixes
)


class ProgressIndicator:
    """
    A simple spinner progress indicator with timing information that respects logging levels.
    """
    def __init__(self, message, logger=None):
        self.message = message
        self.start_time = time.time()
        self._spinner_cycle = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
        self._stop_event = threading.Event()
        self._thread = None
        self.logger = logger or logging.getLogger(__name__)
        
    def _get_elapsed(self):
        return f"{time.time() - self.start_time:.1f}s"
        
    def _animate(self):
        # Only show animation if logging level is INFO or lower
        if self.logger.getEffectiveLevel() <= logging.INFO:
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
        
        # Clear the progress line if we were showing animation
        if self.logger.getEffectiveLevel() <= logging.INFO:
            sys.stdout.write("\r" + " " * 100)  # Clear line
            sys.stdout.write(f"\r{symbol} [{elapsed}] {message}\n")
            sys.stdout.flush()
        
        # Log the final status
        log_level = logging.INFO if success else logging.WARNING
        self.logger.log(log_level, f"Completed ({elapsed}): {message}")
        
        return elapsed
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        self.stop(success=success)

class RobopointGPU(Robopoint_Base):
    """
    RoboPoint GPU implementation using Docker for inference.
    """

    def __init__(
        self,
        model_id: str = "robopoint-v1-vicuna-v1.5-13b",
        docker_image: str = f"{DEFAULT_DOCKER_REPO}",
        auto_pull: bool = True,
        server_port: int = 10001,
        default_instruction: str = "In the image, identify and pinpoint several key locations. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image.",
        verbosity: str = "warning",
        **kwargs
    ):
        """
        Initialize the RoboPoint GPU implementation.
        
        Args:
            model_id: The model ID to use for inference.
            docker_image: The Docker image to use for inference.
            auto_pull: Whether to automatically pull the Docker image if it doesn't exist.
            server_port: Port number for the controller server (default: 10001)
            default_instruction: Default instruction to use when no text_instruction is provided
            verbosity: Logging level ('debug', 'info', 'warning', 'error', 'critical')
            **kwargs: Additional arguments to pass to the base implementation.
        """
        # Configure logging based on verbosity
        verbosity = verbosity.lower()
        log_levels = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }
        log_level = log_levels.get(verbosity, logging.WARNING)
        logger.setLevel(log_level)  # Just set the level, format is already configured
        
        super().__init__()
        self.model_id = model_id
        self.docker_image = docker_image
        self.auto_pull = auto_pull
        self.controller_url = f"http://localhost:{server_port}"  # Construct URL from server_port
        self.default_instruction = default_instruction
        self.worker_addr = None  # Add worker address cache
        
        # Print welcome message
        logging.info("✨ InferX - RoboPoint Model ✨")
        
        # Check if Docker is available
        with ProgressIndicator("Checking Docker availability", logger):
            self.docker_available = self._check_docker_available()
        
        # Check if the Docker image exists and pull it if needed
        if self.docker_available and auto_pull and not self._check_docker_image_exists():
            with ProgressIndicator(f"Pulling InferX Optimized RoboPoint Docker image", logger):
                self._pull_docker_image()

        # Run docker image and wait for server
        self._run_docker_container()
        if not self.wait_for_server():
            raise RuntimeError("Failed to start RoboPoint server")

    def _check_docker_available(self) -> bool:
        """
        Check if Docker is available on the system.
        
        Returns:
            bool: True if Docker is available, False otherwise.
        """
        try:
            result = subprocess.run(
                ["docker", "info"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True
            )
            if result.returncode == 0:
                logger.info("Docker is available.")
                return True
            else:
                logger.warning(f"Docker is not available: {result.stderr}")
                return False
        except Exception as e:
            logger.warning(f"Error checking Docker availability: {e}")
            return False
    
    def _check_docker_image_exists(self) -> bool:
        """
        Check if the Docker image exists locally.
        
        Returns:
            bool: True if the image exists, False otherwise.
        """
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", self.docker_image],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"Error checking Docker image: {e}")
            return False
    
    def _pull_docker_image(self) -> bool:
        """
        Pull the Docker image from Docker Hub.
        
        Returns:
            bool: True if the image was pulled successfully, False otherwise.
        """
        try:
            with ProgressIndicator(f"Pulling RoboPoint Docker image {self.docker_image}", logger).start() as progress:
                result = subprocess.run(
                    ["docker", "pull", self.docker_image],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    text=True
                )
            
            if result.returncode == 0:
                logger.info(f"Docker image {self.docker_image} pulled successfully.")
                return True
            else:
                logger.warning(f"Failed to pull Docker image: {result.stderr}")
                return False
        except Exception as e:
            logger.warning(f"Error pulling Docker image: {e}")
            return False
    
    def install_dependencies(self) -> bool:
        """
        Install dependencies for the RoboPoint GPU implementation.
        This primarily involves ensuring the Docker image is available.
        
        Returns:
            bool: True if dependencies were installed successfully, False otherwise.
        """
        
        # Check if Docker is available
        if not self.docker_available:
            logger.warning("Docker is not available. Cannot install dependencies.")
            return False
        
        # Check if the Docker image exists and pull it if needed
        if not self._check_docker_image_exists():
            with ProgressIndicator(f"Pulling optimized RoboPoint model", logger):
                if self._pull_docker_image():
                    return True
                else:
                    print("❌ Failed to install dependencies.")
                    return False
        else:
            return True
    
    def _run_docker_container(self):
        """
        Run the RoboPoint server image.
        """
        # Check for existing container running the same image
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"ancestor={self.docker_image}", "--format", "{{.ID}}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.stdout.strip():
                logger.info("RoboPoint container is already running")
                return
        except Exception as e:
            logger.warning(f"Error checking for existing containers: {str(e)}")

        # Create the Docker command with expanded home directory path
        huggingface_cache = os.path.expanduser("~/.cache/huggingface/hub")
        cmd = [
            "docker", "run", "--rm",
            "--gpus", "all",
            "--network", "host",
            "-v", f"{huggingface_cache}:/models",
            self.docker_image,
        ]
        
        logger.info(f"Running Docker command: {' '.join(cmd)}")
        
        with ProgressIndicator("Starting RoboPoint GPU server", logger):
            try:
                # Start container in background
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True  # Detach from parent process
                )
                logger.info("Docker container started in background")
            except Exception as e:
                logger.error(f"Failed to start Docker container: {str(e)}")
                raise RuntimeError("Failed to start Docker container") from e

    def wait_for_server(self, timeout=30, interval=2) -> bool:
        """
        Wait for the RoboPoint server to be running and get worker address.
        
        Args:
            timeout: Maximum time to wait in seconds
            interval: Time between retries in seconds
            
        Returns:
            bool: True if server is running and responding, False otherwise
        """
        with ProgressIndicator("Waiting for RoboPoint server to start", logger):
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                try:
                    # Check if models are available
                    response = requests.post(f"{self.controller_url}/list_models")
                    if response.status_code == 200:
                        available_models = response.json().get("models", [])
                        if available_models and self.model_id in available_models:
                            # Get and cache worker address
                            response = requests.post(
                                f"{self.controller_url}/get_worker_address",
                                json={"model": self.model_id}
                            )
                            if response.status_code == 200:
                                self.worker_addr = response.json()["address"]
                                if self.worker_addr:
                                    return True
                except requests.exceptions.RequestException:
                    pass
                
                time.sleep(interval)
            
            return False

    def inference(
        self,
        image_path,
        text_instruction=None,
        output=None,
        temperature=0.2,
        top_p=0.7,
        max_new_tokens=512,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run inference on an image using the RoboPoint model.
        
        Args:
            text_instruction: The instruction to give to the model.
            image_path: The path to the image to run inference on.
            output: The path to save the output image to. If None, no image will be saved.
            temperature: Sampling temperature (default: 0.2)
            top_p: Top p sampling parameter (default: 0.7)
            max_new_tokens: Maximum number of tokens to generate (default: 512)
            **kwargs: Additional arguments to pass to the model.
            
        Returns:
            Dict[str, Any]: The inference results, including keypoints and raw response.
            
        Raises:
            RuntimeError: If Docker is not available or if there are issues with the Docker image or inference.
        """       
 
        # Check if Docker is available
        if not self.docker_available:
            raise RuntimeError("Docker is not available. Cannot run RoboPoint GPU implementation.")
        
        # Check if we have a valid worker address
        if not self.worker_addr:
            raise RuntimeError("No worker address available. Server may not be running properly.")
        
        # Set default prompt if none provided
        if text_instruction is None:
            text_instruction = self.default_instruction
        else:
            text_instruction = self.default_instruction + "\n" + text_instruction
        
        # Replace prints with logger
        logger.info(f"📷 Processing image: {os.path.basename(image_path)}")
        logger.info(f"💬 Instruction: {text_instruction}")
        if output:
            logger.info(f"📁 Output will be saved to: {output}")
        else:
            logger.info("📝 No output file will be saved (output parameter not specified)")
        
        try:
            # Encode image to base64
            with Image.open(image_path) as img:
                # Convert to RGB if not already
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Save to bytes
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                image_b64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Get conversation template based on model name
            if 'vicuna' in self.model_id.lower():
                conv = conv_templates["vicuna_v1"].copy()
            elif "llama" in self.model_id.lower():
                conv = conv_templates["llava_llama_2"].copy()
            elif "mistral" in self.model_id.lower():
                conv = conv_templates["mistral_instruct"].copy()
            elif "mpt" in self.model_id.lower():
                conv = conv_templates["mpt"].copy()
            else:
                conv = conv_templates["llava_v1"].copy()
            
            # Add image token if not present
            if DEFAULT_IMAGE_TOKEN not in text_instruction:
                if 'llava' in self.model_id.lower():
                    text_instruction = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + text_instruction
                else:
                    text_instruction = DEFAULT_IMAGE_TOKEN + '\n' + text_instruction
            
            # Add to conversation
            conv.append_message(conv.roles[0], text_instruction)
            conv.append_message(conv.roles[1], None)
            
            # Get prompt and stop string
            final_prompt = conv.get_prompt()
            stop_str = conv.sep if conv.sep_style in ["SINGLE", "MPT"] else conv.sep2
            
            # Prepare request payload
            payload = {
                "model": self.model_id,
                "prompt": final_prompt,
                "images": [image_b64],
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "stop": stop_str
            }
            
            # Replace print with logger
            logger.info(f"Sending request to worker at {self.worker_addr}")
            
            # Make request to worker using cached address
            with ProgressIndicator("Generating points from image...", logger):
                response = requests.post(
                    f"{self.worker_addr}/worker_generate_stream",
                    json=payload,
                    stream=True,
                    headers={"User-Agent": "RoboPoint Client"}
                )
                
                # Handle streaming response
                full_output = ""
                for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                    if chunk:
                        data = json.loads(chunk.decode())
                        if data["error_code"] == 0:
                            # Print only the new text (remove the prompt)
                            output_text = data["text"][len(final_prompt):].strip()
                            # Use print for real-time display and logger for permanent record
                            # print(output_text, end="\r")
                            full_output = output_text  # Save the last output
                        else:
                            error_msg = f"Error: {data['text']} (code: {data['error_code']})"
                            logger.error(error_msg)
                            raise RuntimeError(error_msg)
            
            # Parse coordinates from the output
            pattern = r"\(([0-9.]+),\s*([0-9.]+)\)"
            matches = re.findall(pattern, full_output)
            keypoints = [(float(x), float(y)) for x, y in matches]
            
            # If output path provided, create visualization
            if output and keypoints:
                with Image.open(image_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    draw = ImageDraw.Draw(img)
                    width, height = img.size
                    
                    # Draw each coordinate as a dot
                    for x, y in keypoints:
                        px = int(x * width)
                        py = int(y * height)
                        radius = 5  # Default dot size
                        draw.ellipse(
                            [(px - radius, py - radius), (px + radius, py + radius)],
                            fill="red"  # Default color
                        )
                    
                    img.save(output)

            logger.info("✨ RoboPoint Inference Summary:")
            logger.info(f"   • Model: {self.model_id}")
            logger.info(f"   • Device: GPU (Docker)")
            logger.info(f"   • Keypoints detected: {len(keypoints)}")
            if output:
                logger.info(f"   • Output visualization: {output}")
            else:
                logger.info(f"   • No output image saved")
            
            return {
                "status": "success",
                "keypoints": keypoints,
                "raw_response": full_output
            }
            
        except Exception as e:
            raise RuntimeError(f"Error during inference: {str(e)}")
        