import torch
import time
import random
import atexit
import signal
from pathlib import Path

class Clip_GPU:
    def __init__(self):
        print(f"\n✨ InferX - CLIP Model (GPU) ✨")
        print(f"🚀 NVIDIA GPU Detected - Using Direct PyTorch Implementation")
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please use CPU implementation.")
        
        print(f"🎯 Device: {torch.cuda.get_device_name(0)}")
        print(f"🔋 CUDA Version: {torch.version.cuda}")
        
        self.device = torch.device("cuda")
        self.model = None
        self.processor = None
        
        # Install dependencies and setup model
        self.install_dependencies()
        self._setup_model()

    def install_dependencies(self):
        """
        Installs the dependencies for the CLIP GPU model.
        """
        import subprocess
        
        print("📦 Installing required packages...")
        
        try:
            subprocess.run([
                "uv", "pip", "install", 
                "transformers", 
                "torch", 
                "torchvision",
                "pillow",
                "accelerate"
            ], check=True, capture_output=True)
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            raise

    def _setup_model(self):
        """Initialize the CLIP model using transformers."""
        try:
            print("🤖 Loading CLIP model...")
            from transformers import CLIPProcessor, CLIPModel
            
            # Load model and processor
            model_name = "openai/clip-vit-base-patch32"
            print(f"📥 Loading {model_name}...")
            
            # Use safetensors to avoid torch.load security issue
            self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            
            # Set model to evaluation mode
            self.model.eval()
            
            print("✅ CLIP model loaded successfully on GPU")
            
            # Print GPU memory usage
            if torch.cuda.is_available():
                print(f"🔋 GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f}GB reserved")
                
        except Exception as e:
            raise RuntimeError(f"Failed to setup CLIP model: {str(e)}")

    def inference(self, image_paths, text_queries=[]):
        """Run inference using CLIP model.
        
        Args:
            image_paths (List[str]): List of paths to images
            text_queries (List[str]): List of text queries to match against
            
        Returns:
            list: List of dictionaries with ranking results
        """
        if not text_queries:
            raise ValueError("text_queries cannot be empty")
            
        from PIL import Image
        import torch.nn.functional as F
        
        print(f"🔍 Processing {len(image_paths)} images with {len(text_queries)} text queries...")
        
        # Load images
        images = []
        valid_image_paths = []
        
        for img_path in image_paths:
            try:
                image = Image.open(img_path).convert('RGB')
                images.append(image)
                valid_image_paths.append(img_path)
            except Exception as e:
                print(f"Warning: Could not load image {img_path}: {e}")
        
        if not images:
            raise ValueError("No valid images found")
        
        results = []
        
        with torch.no_grad():
            # Process images
            image_inputs = self.processor(images=images, return_tensors="pt", padding=True)
            image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
            
            # Get image embeddings
            image_features = self.model.get_image_features(**image_inputs)
            image_features = F.normalize(image_features, p=2, dim=1)
            
            # Process each text query
            for text_query in text_queries:
                # Process text
                text_inputs = self.processor(text=[text_query], return_tensors="pt", padding=True)
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                
                # Get text embeddings
                text_features = self.model.get_text_features(**text_inputs)
                text_features = F.normalize(text_features, p=2, dim=1)
                
                # Calculate similarities
                similarities = torch.mm(text_features, image_features.T).squeeze(0)
                
                # Create matches list with scores
                matches_list = []
                for idx, sim_score in enumerate(similarities):
                    match_dict = {
                        "image_path": valid_image_paths[idx],
                        "score": f"{sim_score.item():.4f}"
                    }
                    matches_list.append(match_dict)
                
                # Sort by score (highest first)
                matches_list.sort(key=lambda x: float(x["score"]), reverse=True)
                
                result = {text_query: matches_list}
                results.append(result)
        
        return results

    def _cleanup(self):
        """Cleanup GPU memory."""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            if hasattr(self, 'processor') and self.processor is not None:
                del self.processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("🧹 GPU memory cleaned up")
        except Exception:
            pass

   