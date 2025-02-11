from ._base import Deepseek_R1_Base
from typing import List, Dict, Generator, Optional, Union
import subprocess
import threading
import time
import requests
import socket
import psutil
from openai import OpenAI

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
        self._wait_for_server()
        
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
        """Run vLLM server in a thread."""
        cmd = [
            "uv",
            "run",
            "vllm",
            "serve",
            self.model_name,
            "--enable-reasoning",
            "--reasoning-parser",
            "deepseek_r1",
            "--port",
            str(self.port)
        ]
        subprocess.run(cmd)

    def __del__(self):
        """Cleanup on deletion."""
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