from ._base import Clip_Base
import os
import subprocess
import time
from PIL import Image
import shutil

class Clip_Orin_Nano(Clip_Base):
    def __init__(self, model_name="openai/clip-vit-large-patch14-336", container_tag=None):
        """
        Initializes CLIP model on Orin Nano using the clip_trt container.
        
        Args:
            model_name (str): Name of the CLIP model to use (from HuggingFace)
            container_tag (str): Version tag for the clip_trt container. If None, will use autotag
        """
        self.model_name = model_name
        self._setup_container(container_tag)
        
    def _setup_container(self, container_tag):
        """
        Ensures the clip_trt container is available and properly set up.
        Will attempt to install jetson-containers if not present.
        """
        # First check if jetson-containers is installed
        if not os.path.exists(os.path.expanduser("~/jetson-containers")):
            print("Installing jetson-containers...")
            subprocess.run([
                "git", "clone", "https://github.com/dusty-nv/jetson-containers",
                os.path.expanduser("~/jetson-containers")
            ], check=True)
            subprocess.run([
                "bash", os.path.expanduser("~/jetson-containers/install.sh")
            ], check=True)
        
        # Get the appropriate container tag if not specified
        if container_tag is None:
            try:
                result = subprocess.run(
                    ["jetson-containers", "autotag", "clip_trt"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                self.container_tag = result.stdout.strip()
            except subprocess.CalledProcessError:
                raise RuntimeError("Failed to determine compatible clip_trt container tag")
        else:
            self.container_tag = container_tag
            
        # Build/pull the container
        try:
            subprocess.run([
                "jetson-containers", "build", "clip_trt"
            ], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to build clip_trt container: {e}")

    def _load_images(self, image_input):
        """
        Loads images from paths and returns PIL Image objects.
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

        images = []
        valid_paths = []
        for path in image_paths:
            if os.path.exists(path):
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(img)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"Error loading image {path}: {e}")
        return images, valid_paths

    def inference(self, image_paths, classes=[]):
        """
        Runs CLIP inference using the clip_trt container.
        
        Args:
            image_paths: String or list of image paths
            classes: List of text classes to compare against
            
        Returns:
            List of dictionaries containing predictions for each image
        """
        images, valid_paths = self._load_images(image_paths)
        if not images:
            return {"error": "No valid images found"}

        # Create a temporary directory for image processing
        tmp_dir = "/tmp/clip_trt_input"
        os.makedirs(tmp_dir, exist_ok=True)
        
        # Save images with their original names to preserve paths
        temp_paths = []
        for i, (img, orig_path) in enumerate(zip(images, valid_paths)):
            temp_path = os.path.join(tmp_dir, os.path.basename(orig_path))
            img.save(temp_path)
            temp_paths.append(temp_path)

        try:
            # Run inference using clip_trt's Python module
            cmd = [
                "jetson-containers", "run",
                "-v", f"{tmp_dir}:/workspace/images",
                f"clip_trt:{self.container_tag}",
                "python3", "-m", "clip_trt",
                "--model", self.model_name,
                "--use_tensorrt", "True",
                "--inputs"
            ] + [f"/workspace/images/{os.path.basename(p)}" for p in valid_paths] + ["--inputs"] + classes

            output = subprocess.check_output(cmd, text=True)
            
            # Parse results - clip_trt outputs similarity scores
            predictions = []
            for line in output.strip().split("\n"):
                if "similarity scores:" in line.lower():
                    scores = [float(x) for x in line.split(":")[1].strip().split()]
                    best_idx = max(range(len(scores)), key=scores.__getitem__)
                    predictions.append({
                        "best_class": classes[best_idx],
                        "probability": scores[best_idx]
                    })
            
            return predictions

        finally:
            # Cleanup
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def train(self):
        pass
