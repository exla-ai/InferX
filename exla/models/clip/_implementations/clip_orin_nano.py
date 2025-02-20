from ._base import Clip_Base
import os
import subprocess

class Clip_Orin_Nano(Clip_Base):
    def __init__(self, model_name="openai/clip-vit-large-patch14-336", container_tag="36.3.0"):
        """
        Initializes CLIP model on Orin Nano using the clip_trt container.
        
        Args:
            model_name (str): Name of the CLIP model to use (from HuggingFace)
            container_tag (str): Version tag for the clip_trt container. Defaults to 36.3.0
        """
        self.model_name = model_name
        self.container_tag = container_tag
        self._setup_container()
        self.install_dependencies()

    def install_dependencies(self):
        """
        Installs the dependencies for the CLIP model on Orin Nano.
        """
        subprocess.run([
            "uv", "pip", "install", "-r", "requirements/requirements_orin_nano.txt"
        ], check=True)

    def _setup_container(self):
        """
        Ensures the clip_trt container is available.
        """
        try:
            # Check if container exists
            result = subprocess.run([
                "docker", "images", "-q", f"clip_trt:{self.container_tag}"
            ], capture_output=True, text=True, check=True)
            
            if not result.stdout.strip():
                print(f"Pulling clip_trt:{self.container_tag} container...")
                # Pull the container
                subprocess.run([
                    "docker", "run", "--runtime", "nvidia", "-it", "--rm", "--network=host",
                    f"clip_trt:{self.container_tag}", "echo", "Container setup complete"
                ], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to setup clip_trt container: {e}")

    def _load_images(self, image_input):
        """
        Loads images from paths and returns PIL Image objects.
        """
        from PIL import Image
        import time
        
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
        import shutil
        
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
                "docker", "run", "--runtime", "nvidia", "--rm", "--network=host",
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

    