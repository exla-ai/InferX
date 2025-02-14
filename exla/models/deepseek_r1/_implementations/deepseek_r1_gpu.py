from ._base import Deepseek_R1_Base
from typing import List, Dict, Generator, Optional, Union
import subprocess
import threading
import time
import requests
import socket
import psutil
from openai import OpenAI
from pathlib import Path

class Deepseek_R1_GPU(Deepseek_R1_Base):
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", port=8000):
        print("Initializing Deepseek R1 model for GPU...")
        self.model_name = model_name
        self.port = port
        
        # Clean up any existing process using the port
        self._cleanup_port()
        
        # Start vLLM server in a separate thread
        self._server_thread = threading.Thread(
            target=self._run_server,
            daemon=True
        )
        self._server_thread.start()
        
        # Wait for server to be ready
        # self._wait_for_server()
        
        self._client = OpenAI(
            api_key="EMPTY",
            base_url=f"http://localhost:{self.port}/v1"
        )

    def _verify_port_free(self):
        """Check if port is free."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', self.port))
                s.close()
                return True
            except OSError:
                return False

    def _cleanup_port(self):
        """Clean up any process using the port."""
        if not self._verify_port_free():
            print(f"Port {self.port} is in use, cleaning up...")
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    for conn in proc.connections('inet'):
                        if conn.laddr.port == self.port:
                            proc.terminate()
                            proc.wait(timeout=5)
                            break
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                    continue
            
            # Double check port is free
            if not self._verify_port_free():
                raise RuntimeError(f"Failed to free port {self.port}")

    def _wait_for_server(self, timeout=120):
        """Wait for server to be ready."""
        print("Waiting for server to start...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://localhost:{self.port}/v1/models")
                if response.status_code == 200:
                    print("Server is ready!")
                    return
            except requests.exceptions.ConnectionError:
                time.sleep(1)
                continue
        raise RuntimeError(f"Server did not start within {timeout} seconds")

    def _run_server(self):
        """Run vLLM server in a Docker container."""
        try:
            print("Starting vLLM server in Docker...")
            cmd = [
                "sudo", "docker", "run",
                "--gpus", "all",  # Simplified GPU access
                "-v", f"{str(Path.home())}/.cache/huggingface:/root/.cache/huggingface",
                "-p", f"{self.port}:8000",
                "--ipc=host",
                "--rm",  # Remove container when stopped
                "-d",    # Run in detached mode
                "vllm/vllm-openai:latest",
                "--model", self.model_name
            ]
            
            # First check if nvidia-container-toolkit is installed
            check = subprocess.run(["nvidia-smi"], capture_output=True)
            if check.returncode != 0:
                raise RuntimeError("NVIDIA drivers not found. Please install NVIDIA drivers.")
            
            # Check if Docker has GPU access
            check = subprocess.run(["sudo", "docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:11.8.0-base-ubuntu22.04", "nvidia-smi"], capture_output=True)
            if check.returncode != 0:
                print("GPU access not configured in Docker. Installing NVIDIA Container Toolkit...")
                subprocess.run([
                    "sudo", "apt-get", "update"
                ], check=True)
                subprocess.run([
                    "sudo", "apt-get", "install", "-y", "nvidia-container-toolkit"
                ], check=True)
                subprocess.run([
                    "sudo", "nvidia-ctk", "runtime", "configure", "--runtime=docker"
                ], check=True)
                subprocess.run([
                    "sudo", "systemctl", "restart", "docker"
                ], check=True)
            
            # Now try to run vLLM
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to start vLLM container: {result.stderr}")
            
            # Store container ID for cleanup
            self._container_id = result.stdout.strip()
            
        except Exception as e:
            raise RuntimeError(f"Failed to start vLLM server: {str(e)}")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            if hasattr(self, '_container_id') and self._container_id:
                subprocess.run(["sudo", "docker", "stop", self._container_id], 
                             capture_output=True)
        except:
            pass
        self._cleanup_port()

    def chat(self):
        """Start OpenWebUI chat interface."""
        try:
            # Start OpenWebUI with our vLLM server
            subprocess.run([
                "sudo", "docker", "run", "-d",
                "-p", "3000:8080",  # OpenWebUI port
                "-e", f"OPENAI_API_BASE=http://localhost:{self.port}/v1",
                "-v", "open-webui:/app/backend/data",
                "--name", "open-webui",
                "--restart", "always",
                "ghcr.io/open-webui/open-webui:main"
            ])
            print(f"\nChat interface available at: http://localhost:3000")
            
        except Exception as e:
            print(f"Failed to start chat interface: {e}")