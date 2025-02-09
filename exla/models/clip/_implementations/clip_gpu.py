import subprocess
import time
import os
from pathlib import Path
import atexit
import socket
import psutil
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
    
    def _verify_port_free(self):
        """Verify if port 51000 is actually free."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', 51000))
                s.close()
                return True
            except OSError:
                return False

    def _cleanup_port(self):
        """Clean up any process using port 51000."""
        print("Cleaning up port 51000...")
        try:
            # Find and kill any process using port 51000
            for proc in psutil.process_iter():
                try:
                    # Get connections for this process
                    connections = proc.connections()
                    for conn in connections:
                        if conn.laddr.port == 51000:
                            print(f"Found process {proc.pid} using port 51000, killing it...")
                            proc.kill()  # Use SIGKILL for immediate termination
                            proc.wait()  # Wait for process to die
                            break
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue  # Skip if we can't check this process
            
            # Verify port is actually free
            time.sleep(1)  # Wait a bit longer for cleanup
            if self._verify_port_free():
                print("Port 51000 is now free")
                return True
            
            print("Port still in use after cleanup attempt")
            return False
            
        except Exception as e:
            print(f"Error during port cleanup: {str(e)}")
            return False

    def start(self):
        if self._is_running:
            return
            
        try:
            # Clean up any existing process on the port first
            if not self._cleanup_port():
                raise RuntimeError("Failed to clean up port 51000 after multiple attempts")
            
            # Double check port is free
            if not self._verify_port_free():
                raise RuntimeError("Port 51000 is still in use after cleanup")
            
            print("Starting CLIP server...")
            # Start server with PyTorch backend
            self._process = subprocess.Popen(
                ["python", "-m", "clip_server"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            time.sleep(5)
            
            # Check if server started successfully
            if self._process.poll() is not None:
                stdout, stderr = self._process.communicate()
                print("Server stdout:", stdout)
                print("Server stderr:", stderr)
                raise RuntimeError(f"Failed to start CLIP server. Error: {stderr}")
            
            self._is_running = True
            print("CLIP server started successfully")
                
        except Exception as e:
            if self._process:
                self._process.terminate()
                self._process = None
            raise RuntimeError(f"Failed to start CLIP server: {str(e)}")
    
    def stop(self):
        """Stop the server and cleanup resources."""
        if self._process:
            try:
                # Try graceful shutdown first
                self._process.terminate()
                for _ in range(3):  # Wait up to 3 seconds
                    if self._process.poll() is not None:
                        break
                    time.sleep(1)
                
                # If still running, force kill
                if self._process.poll() is None:
                    self._process.kill()
                    self._process.wait(timeout=1)
                
                # Clean up the port
                self._cleanup_port()
                
            except:
                pass  # Ignore cleanup errors
            finally:
                self._process = None
                self._is_running = False

class Clip_GPU:
    def __init__(self):
        self._server = ClipServer.get_instance()
        self._start_server()
        self._setup_client()
        # Register cleanup on both normal exit and keyboard interrupt
        atexit.register(self._cleanup)
        import signal
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\nCleaning up CLIP server...")
        self._cleanup()
        import sys
        sys.exit(0)
        
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
        try:
            if hasattr(self, '_client'):
                del self._client
            if hasattr(self, '_server'):
                self._server.stop()
        except:
            pass  # Ensure cleanup continues even if there are errors

    def train(self):
        pass    
