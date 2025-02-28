"""
RoboPoint model implementation for GPU.
"""

import os
import sys
import logging
import subprocess
import tempfile
import json
import shutil
from typing import Dict, List, Optional, Tuple, Union, Any
from PIL import Image

from ._base import Robopoint_Base
import torch

logger = logging.getLogger(__name__)

# Default Docker Hub repository for the RoboPoint image
DEFAULT_DOCKER_REPO = "viraatdas/robopoint-gpu"
DEFAULT_DOCKER_TAG = "latest"

class RobopointGPU(Robopoint_Base):
    """
    RoboPoint GPU implementation using Docker for inference.
    """

    def __init__(
        self,
        model_id: str = "wentao-yuan/robopoint-v1-vicuna-v1.5-13b",
        docker_image: str = f"{DEFAULT_DOCKER_REPO}:{DEFAULT_DOCKER_TAG}",
        auto_pull: bool = True,
        **kwargs
    ):
        """
        Initialize the RoboPoint GPU implementation.
        
        Args:
            model_id: The model ID to use for inference.
            docker_image: The Docker image to use for inference.
            auto_pull: Whether to automatically pull the Docker image if it doesn't exist.
            **kwargs: Additional arguments to pass to the base implementation.
        """
        super().__init__()
        self.model_id = model_id
        self.docker_image = docker_image
        self.auto_pull = auto_pull
        
        # Check if Docker is available
        self.docker_available = self._check_docker_available()
        
        # Check if the Docker image exists and pull it if needed
        if self.docker_available and auto_pull and not self._check_docker_image_exists():
            self._pull_docker_image()
        
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
            logger.info(f"Pulling Docker image {self.docker_image} from Docker Hub...")
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
    
    def _build_docker_image(self) -> bool:
        """
        Build the Docker image if it doesn't exist.
        
        Returns:
            bool: True if the image was built successfully, False otherwise.
        """
        if self._check_docker_image_exists():
            logger.info(f"Docker image {self.docker_image} already exists.")
            return True
        
        # Get the absolute path to the Dockerfile directory
        # Assuming the Dockerfile is in docker/models/robopoint/gpu relative to the project root
        try:
            # Try to find the project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = current_dir
            
            # Navigate up until we find the docker directory or reach the filesystem root
            while not os.path.exists(os.path.join(project_root, "docker")) and project_root != "/":
                project_root = os.path.dirname(project_root)
            
            if not os.path.exists(os.path.join(project_root, "docker")):
                logger.warning("Could not find docker directory in project root.")
                return False
            
            dockerfile_dir = os.path.join(project_root, "docker", "models", "robopoint", "gpu")
            
            if not os.path.exists(dockerfile_dir):
                logger.warning(f"Dockerfile directory not found: {dockerfile_dir}")
                return False
            
            logger.info(f"Building Docker image {self.docker_image} from {dockerfile_dir}")
            
            result = subprocess.run(
                ["docker", "build", "-t", self.docker_image, dockerfile_dir],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Docker image {self.docker_image} built successfully.")
                return True
            else:
                logger.warning(f"Failed to build Docker image: {result.stderr}")
                return False
        except Exception as e:
            logger.warning(f"Error building Docker image: {e}")
            return False
    
    def push_docker_image(self, repository: str = DEFAULT_DOCKER_REPO, tag: str = DEFAULT_DOCKER_TAG) -> bool:
        """
        Push the Docker image to Docker Hub.
        
        Args:
            repository: The Docker Hub repository to push to.
            tag: The tag to use for the image.
            
        Returns:
            bool: True if the image was pushed successfully, False otherwise.
        """
        if not self.docker_available:
            logger.warning("Docker is not available. Cannot push image.")
            return False
        
        # Check if the image exists locally
        if not self._check_docker_image_exists():
            logger.warning(f"Docker image {self.docker_image} does not exist locally. Building it first...")
            if not self._build_docker_image():
                logger.warning("Failed to build Docker image. Cannot push.")
                return False
        
        # Tag the image with the repository and tag
        target_image = f"{repository}:{tag}"
        try:
            logger.info(f"Tagging Docker image {self.docker_image} as {target_image}...")
            tag_result = subprocess.run(
                ["docker", "tag", self.docker_image, target_image],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True
            )
            
            if tag_result.returncode != 0:
                logger.warning(f"Failed to tag Docker image: {tag_result.stderr}")
                return False
            
            # Push the image to Docker Hub
            logger.info(f"Pushing Docker image {target_image} to Docker Hub...")
            push_result = subprocess.run(
                ["docker", "push", target_image],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True
            )
            
            if push_result.returncode == 0:
                logger.info(f"Docker image {target_image} pushed successfully.")
                return True
            else:
                logger.warning(f"Failed to push Docker image: {push_result.stderr}")
                return False
        except Exception as e:
            logger.warning(f"Error pushing Docker image: {e}")
            return False
    
    def install_dependencies(self) -> bool:
        """
        Install dependencies for the RoboPoint GPU implementation.
        This primarily involves ensuring the Docker image is available.
        
        Returns:
            bool: True if dependencies were installed successfully, False otherwise.
        """
        logger.info("Installing dependencies for RoboPoint GPU implementation...")
        
        # Check if Docker is available
        if not self.docker_available:
            logger.warning("Docker is not available. Cannot install dependencies.")
            return False
        
        # Check if the Docker image exists and pull it if needed
        if not self._check_docker_image_exists():
            logger.info(f"Docker image {self.docker_image} not found locally. Pulling from Docker Hub...")
            if self._pull_docker_image():
                logger.info("Dependencies installed successfully.")
                return True
            else:
                logger.warning("Failed to pull Docker image. Trying to build locally...")
                if self._build_docker_image():
                    logger.info("Dependencies installed successfully (built locally).")
                    return True
                else:
                    logger.error("Failed to install dependencies.")
                    return False
        else:
            logger.info("Dependencies already installed.")
            return True
    
    def inference(
        self,
        image_path,
        text_instruction=None,
        output=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run inference on an image using the RoboPoint model.
        
        Args:
            image_path: The path to the image to run inference on.
            text_instruction: The instruction to give to the model.
            output: The path to save the output image to.
            **kwargs: Additional arguments to pass to the model.
            
        Returns:
            Dict[str, Any]: The inference results, including keypoints and raw response.
        """
        # Check if Docker is available
        if not self.docker_available:
            logger.warning("Docker is not available. Falling back to base implementation.")
            return super().inference(image_path, text_instruction, output)
        
        # Check if the Docker image exists and pull or build it if needed
        if not self._check_docker_image_exists():
            if self.auto_pull:
                if not self._pull_docker_image():
                    logger.warning("Failed to pull Docker image. Trying to build it locally...")
                    if not self._build_docker_image():
                        logger.warning("Failed to build Docker image. Falling back to base implementation.")
                        return super().inference(image_path, text_instruction, output)
            else:
                if not self._build_docker_image():
                    logger.warning("Failed to build Docker image. Falling back to base implementation.")
                    return super().inference(image_path, text_instruction, output)
        
        # Create a temporary directory with full permissions for Docker output
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Give full permissions to the temp directory
                os.chmod(temp_dir, 0o777)
                
                # Handle the input image
                input_path = image_path
                
                # Set the output path
                if output is None:
                    temp_output_path = os.path.join(temp_dir, "output.jpg")
                else:
                    temp_output_path = output
                    # Ensure the output directory exists
                    os.makedirs(os.path.dirname(os.path.abspath(temp_output_path)), exist_ok=True)
                
                # Set the default instruction if none is provided
                if text_instruction is None:
                    text_instruction = "In the image, identify and pinpoint several key locations. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image."
                
                # Run the Docker container for inference
                try:
                    # Create the Docker command
                    cmd = [
                        "docker", "run", "--rm", "--gpus", "all",
                        "-v", f"{os.path.abspath(os.path.dirname(input_path))}:/app/input",
                        "-v", f"{os.path.abspath(temp_dir)}:/app/output",
                        self.docker_image,
                        "inference",
                        f"/app/input/{os.path.basename(input_path)}",
                        text_instruction,
                        f"/app/output/{os.path.basename(temp_output_path) if output else 'output.jpg'}"
                    ]
                    
                    logger.info(f"Running Docker command: {' '.join(cmd)}")
                    
                    result = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=False,
                        text=True
                    )
                    
                    if result.returncode != 0:
                        logger.warning(f"Docker inference failed: {result.stderr}")
                        logger.warning("Falling back to base implementation.")
                        return super().inference(image_path, text_instruction, output)
                    
                    # Check if the output file exists in the temp directory
                    temp_output_file = os.path.join(temp_dir, os.path.basename(temp_output_path) if output else "output.jpg")
                    if not os.path.exists(temp_output_file):
                        logger.warning(f"Output file not found: {temp_output_file}")
                        # Check if there's a text file with the keypoints
                        text_output_file = f"{temp_output_file}.txt"
                        if os.path.exists(text_output_file):
                            logger.info(f"Found text output file: {text_output_file}")
                            with open(text_output_file, "r") as f:
                                keypoints_line = f.read().strip()
                                if keypoints_line.startswith("Keypoints:"):
                                    keypoints_str = keypoints_line.replace("Keypoints:", "").strip()
                                    try:
                                        import ast
                                        keypoints = ast.literal_eval(keypoints_str)
                                        return {
                                            "status": "success",
                                            "keypoints": keypoints,
                                            "raw_response": "Generated from text file"
                                        }
                                    except Exception as e:
                                        logger.warning(f"Failed to parse keypoints from text file: {e}")
                        
                        logger.warning("Falling back to base implementation.")
                        return super().inference(image_path, text_instruction, output)
                    
                    # If output path was provided, copy the file from temp directory
                    if output:
                        shutil.copy2(temp_output_file, output)
                    
                    # Parse the output to extract keypoints
                    keypoints = []
                    raw_response = ""
                    
                    for line in result.stdout.splitlines():
                        if line.startswith("Keypoints:"):
                            keypoints_str = line.replace("Keypoints:", "").strip()
                            try:
                                # Parse the keypoints string into a list of tuples
                                # The format is expected to be [(x1, y1), (x2, y2), ...]
                                import ast
                                keypoints = ast.literal_eval(keypoints_str)
                            except Exception as e:
                                logger.warning(f"Failed to parse keypoints: {e}")
                        elif line.startswith("Raw response:"):
                            raw_response = line.replace("Raw response:", "").strip()
                    
                    return {
                        "status": "success",
                        "keypoints": keypoints,
                        "raw_response": raw_response
                    }
                    
                except Exception as e:
                    logger.warning(f"Error running Docker inference: {e}")
                    logger.warning("Falling back to base implementation.")
                    return super().inference(image_path, text_instruction, output)
            except Exception as e:
                logger.warning(f"Error setting up temporary directory: {e}")
                logger.warning("Falling back to base implementation.")
                return super().inference(image_path, text_instruction, output)   
        