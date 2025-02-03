import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os

class CLIP_Orin_Nano:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Initializes a CLIP model on Orin Nano with TensorRT-like optimizations.
        """
        self.device = torch.device("cuda")
        print(f"Using device: {self.device}")

        # Load CLIP model 
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def train(self):
        pass

    def _load_images(self, image_input):
        """
        Handles loading images from a single file path, a list of paths, or a text file containing paths.
        Returns a list of PIL Image objects and their corresponding file names.
        """
        image_paths = []

        if isinstance(image_input, str):
            if image_input.endswith(".txt"):  # If a text file is provided, read image paths
                with open(image_input, "r") as f:
                    image_paths = [line.strip() for line in f.readlines()]
            else:  # Single image path
                image_paths = [image_input]
        elif isinstance(image_input, list):  # List of image paths
            image_paths = image_input

        images = []
        valid_paths = []

        for path in image_paths:
            if os.path.exists(path):
                try:
                    img = Image.open(path).convert("RGB")
                    img.save(f"processed_{os.path.basename(path)}")  # Save a copy
                    images.append(img)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"Error loading image {path}: {e}")

        return images, valid_paths

    def inference(self, image_paths, classes=[]):
        """
        Runs inference on one or multiple images, identifying the most similar class.
        - image_input: Single image path, a list of image paths, or a text file containing paths.
        - classes: List of textual class descriptions to compare against.

        Returns a dictionary mapping image filenames to their predicted best class and probability.
        """
        print("Images can be local paths or urls")
        images, valid_paths = self._load_images(image_input)

        if not images:
            return {"error": "No valid images found."}

        # Process images and text together
        inputs = self.processor(
            text=classes,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Model inference
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # Shape: (num_images, num_classes)
        probs = logits_per_image.softmax(dim=1)  # Convert to probability distribution

        # Create structured output
        results = {}
        for i, path in enumerate(valid_paths):
            best_idx = torch.argmax(probs[i]).item()
            best_class = classes[best_idx] if classes else "Unknown"
            results[path] = {
                "best_class": best_class,
                "probability": probs[i, best_idx].item()
            }

        return results
