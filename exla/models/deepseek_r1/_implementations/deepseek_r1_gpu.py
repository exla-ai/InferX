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

        # Wait for the vLLM server to be ready
        self._wait_for_server()

        self._client = OpenAI(
            api_key="Empty",  # Use a non-empty value (e.g., "token-abc123") if required
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
        """Wait for Exla server to be ready."""
        print("Waiting for Exla server to start...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://localhost:{self.port}/v1/models")
                if response.status_code == 200:
                    print("Exla server is ready!")
                    return
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        raise RuntimeError(f"Server did not start within {timeout} seconds")

    def _run_server(self):
        """Run Exla server in a Docker container."""
        try:
            print("Starting Exla server in Docker...")
            cmd = [
                "sudo", "docker", "run",
                "--gpus", "all",  # Use GPU
                "-v", f"{str(Path.home())}/.cache/huggingface:/root/.cache/huggingface",
                "-p", f"{self.port}:8000",
                "--ipc=host",
                "--rm",  # Remove container when stopped
                "vllm/vllm-openai:latest",
                "--model", self.model_name,
                "--enable-reasoning",
                "--reasoning-parser", "deepseek_r1",
                "--uvicorn-log-level", "debug",
                "--return-tokens-as-token-ids"
            ]

            # First check if nvidia-container-toolkit is installed
            check = subprocess.run(["nvidia-smi"], capture_output=True)
            if check.returncode != 0:
                raise RuntimeError("NVIDIA drivers not found. Please install NVIDIA drivers.")

            # Check if Docker has GPU access
            check = subprocess.run(
                ["sudo", "docker", "run", "--rm", "--gpus", "all",
                 "nvidia/cuda:11.8.0-base-ubuntu22.04", "nvidia-smi"],
                capture_output=True
            )
            if check.returncode != 0:
                print("GPU access not configured in Docker. Installing NVIDIA Container Toolkit...")
                subprocess.run(["sudo", "apt-get", "update"], check=True)
                subprocess.run(["sudo", "apt-get", "install", "-y", "nvidia-container-toolkit"], check=True)
                subprocess.run(["sudo", "nvidia-ctk", "runtime", "configure", "--runtime=docker"], check=True)
                subprocess.run(["sudo", "systemctl", "restart", "docker"], check=True)

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to start Exla container: {result.stderr}")

            # Store container ID for cleanup
            self._container_id = result.stdout.strip()
            print(f"Model server started with container ID: {self._container_id}")

        except Exception as e:
            raise RuntimeError(f"Failed to start Exla server: {str(e)}")
            
    def __del__(self):
        """Cleanup on deletion."""
        try:
            if hasattr(self, '_container_id') and self._container_id:
                subprocess.run(["sudo", "docker", "stop", self._container_id], capture_output=True)
        except Exception:
            pass
        self._cleanup_port()

    def chat(self):
        """Start Open WebUI chat interface using Exla as backend."""
        try:
            # Launch the Open WebUI container with additional environment variables.
            # Adjust the following environment variables as needed:
            # - ENABLE_RAG_WEB_SEARCH and RAG_WEB_SEARCH_ENGINE for RAG features.
            # - ENABLE_OLLAMA_API=false to bypass Ollama-specific functionality.
            # - OPENAI_API_KEY set to a non-empty value (e.g., "token-abc123").
            cmd = [
                "sudo", "docker", "run", "-d",
                "-p", "3000:8080",  # Map container's port 8080 to host port 3000.
                "--add-host", "host.docker.internal:host-gateway",
                "-e", f"OPENAI_API_BASE_URL=http://host.docker.internal:{self.port}/v1",
                "-e", "OPENAI_API_KEY=EMPTY",
                "-e", "ENABLE_OLLAMA_API=false",
                "-e", "ENABLE_RAG_WEB_SEARCH=true",
                "-e", "RAG_WEB_SEARCH_ENGINE=duckduckgo",
                "-v", "open-webui:/app/backend/data",
                "--name", "open-webui",
                "--restart", "always",
                "ghcr.io/open-webui/open-webui:main"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to start Open WebUI container: {result.stderr}")
            print("\nChat interface available at: http://localhost:3000")
        except Exception as e:
            print(f"Failed to start chat interface: {e}")
