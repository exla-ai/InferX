import subprocess
import time
import os
from pathlib import Path
import atexit
import importlib.util
import socket
import signal
import random
from docarray import Document, DocumentArray
from clip_client import Client

class ClipServer:
    _instance = None
    _is_running = False
    _process = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def start(self):
        if self._is_running:
            return
            
        try:
            # Start server with PyTorch backend
            self._process = subprocess.Popen(
                ["python", "-m", "clip_server", "onnx-flow.yml"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            time.sleep(5)
            
            # Check if server started successfully
            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode()
                raise RuntimeError(f"Failed to start CLIP server. Error: {stderr}")
            
            self._is_running = True
                
        except Exception as e:
            if self._process:
                self._process.terminate()
                self._process = None
            raise RuntimeError(f"Failed to start CLIP server: {str(e)}")
    
    def stop(self):
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=3)
            except:
                self._process.kill()
            
            self._process = None
            self._is_running = False

class Clip_GPU:
    def __init__(self):
        self._server = ClipServer.get_instance()
        self._start_server()
        self._setup_client()
        atexit.register(self._cleanup)

    def _start_server(self):
        """Start the CLIP server."""
        try:
            self._server.start()
        except Exception as e:
            raise RuntimeError(f"Failed to start CLIP server: {str(e)}")

    def _setup_client(self):
        """Initialize the CLIP client."""
        try:
            self._client = Client('grpc://localhost:51000')
        except Exception as e:
            raise RuntimeError(f"Failed to connect to CLIP server: {str(e)}")

    def inference(self, image_paths, classes=[]):
        """Run inference using CLIP model.
        
        Args:
            image_paths (List[str]): List of paths to images (local paths or URLs)
            classes (List[str]): List of text classes to match against
            
        Returns:
            dict: Dictionary mapping image paths to their best predictions
        """
        try:
            # Create a DocumentArray for the images
            docs = DocumentArray()
            for img_path in image_paths:
                # Convert local paths to absolute paths
                if not img_path.startswith(('http://', 'https://', 'data:')):
                    img_path = str(Path(img_path).absolute())
                
                # Create a Document with the image and text matches
                doc = Document(uri=img_path)
                doc.matches = [Document(text=c) for c in classes]
                docs.append(doc)
            
            # Use the rank endpoint to get predictions
            results = self._client.rank(docs)
            output = dict()

            for doc, image_path in zip(results, image_paths):
                best_match = max(doc.matches, key=lambda m: m.scores['clip_score'].value)
                best_class = best_match.text
                best_score = f"{best_match.scores['clip_score'].value:.4f}"
                output[image_path] = {
                    "best_class": best_class,
                    "best_score": best_score
                }
                
            return output
            
        except Exception as e:
            raise RuntimeError(f"Inference failed: {str(e)}")

    def _cleanup(self):
        """Clean up resources when the object is destroyed."""
        if hasattr(self, '_server'):
            self._server.stop()

    def train(self):
        pass    
