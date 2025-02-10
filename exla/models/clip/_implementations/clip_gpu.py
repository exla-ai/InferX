import subprocess
import time
import random
import atexit
import signal
from pathlib import Path
from clip_client import Client

class Clip_GPU:
    def __init__(self):
        # Generate random 5 digit port between 10000-65535
        self._port = random.randint(10000, 65535)
        self._container_id = None
        
        # Register cleanup handlers
        atexit.register(self._cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self._start_server()
        self._setup_client()

    def _start_server(self):
        """Start the CLIP server using Docker."""
        try:
            print(f"Starting CLIP server on port {self._port}...")
            
            # Start container and get its ID
            result = subprocess.run(
                [
                    "sudo", "docker", "run",
                    "-d",  # Run in detached mode
                    "--rm",  # Automatically remove container when it exits
                    "-p", f"{self._port}:51000",
                    "-v", f"{str(Path.home())}/.cache:/home/cas/.cache",
                    "--gpus", "all",
                    "jinaai/clip-server:master-tensorrt",
                    "tensorrt-flow.yml"
                ],
                capture_output=True,
                text=True
            )
            
            # Store container ID for cleanup
            self._container_id = result.stdout.strip()
            
            # Give server time to start
            time.sleep(10)
            print("CLIP server started successfully")
            
        except Exception as e:
            self._cleanup()
            raise RuntimeError(f"Failed to start CLIP server: {str(e)}")

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        self._cleanup()
        exit(0)

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
      
        results = []
        
        for doc in ranked_docs:
            result = {}
            matches_list = []
            for match in doc.matches:
                score = match.scores["clip_score"].value
                match_dict = {
                    "image_path": match.uri,
                    "score": f"{score:.4f}"
                }
                matches_list.append(match_dict)
            result[doc.text] = matches_list
            results.append(result)
            
        # Output is a list of dictionaries, each dictionary 
        #contains a text query and a list of image paths 
        # and scores and they are sorted by score
        return results

    def _cleanup(self):
        """Cleanup Docker container on exit."""
        try:
            if self._container_id:
                print("\nCleaning up CLIP server...")
                subprocess.run(["sudo", "docker", "stop", self._container_id], 
                             capture_output=True)
                self._container_id = None
        except:
            pass

   