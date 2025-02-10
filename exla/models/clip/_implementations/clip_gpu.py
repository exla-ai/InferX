import subprocess
import time
import random
import atexit
from pathlib import Path
from clip_client import Client

class Clip_GPU:
    def __init__(self):
        # Generate random 5 digit port between 10000-65535
        self._port = random.randint(10000, 65535)
        self._start_server()
        self._setup_client()
        atexit.register(self._cleanup)
        
    def _start_server(self):
        """Start the CLIP server using Docker."""
        try:
            print(f"Starting CLIP server on port {self._port}...")
            self._process = subprocess.Popen(
                [
                    "sudo", "docker", "run",
                    "--rm",  # Automatically remove container when it exits
                    "-p", f"{self._port}:51000",  # Map container port 51000 to host port
                    "-v", f"{str(Path.home())}/.cache:/home/cas/.cache",
                    "--gpus", "all",
                    "jinaai/clip-server:master-tensorrt",
                    "tensorrt-flow.yml"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            # Give server time to start
            time.sleep(10)  # Increased wait time
            print("CLIP server started successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to start CLIP server: {str(e)}")

    def _setup_client(self):
        """Initialize the CLIP client."""
        try:
            self._client = Client(f'grpc://0.0.0.0:{self._port}')
            time.sleep(2)  # Give client time to connect
        except Exception as e:
            raise RuntimeError(f"Failed to connect to CLIP server: {str(e)}")

    def inference(self, image_paths, text_queries=[]):
        """Run inference using CLIP model.
        
        Args:
            image_paths (List[str]): List of paths to images
            classes (List[str]): List of text classes to match against
            
        Returns:
            dict: Dictionary where each key is a class and value is a list of 
                 ranked image matches with their scores
        """
        from docarray import Document, DocumentArray
        
        # Create documents for images
        docs = DocumentArray()
        for text_query in text_queries:
            doc = Document(text=text_query)
            doc.matches = [Document(uri=img_path) for img_path in image_paths]
            docs.append(doc)

        # Use the rank endpoint
        ranked_docs = self._client.rank(docs)
        print(ranked_docs)
        
        return ranked_docs

    def _cleanup(self):
        """Cleanup Docker container on exit."""
        try:
            if hasattr(self, '_process'):
                subprocess.run(["sudo", "docker", "ps", "-q", "--filter", "ancestor=jinaai/clip-server:master-tensorrt"], 
                             capture_output=True, text=True)
                self._process.terminate()
        except:
            pass

   