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

    def chat(self, 
        messages: List[Dict[str, str]], 
        stream: bool = False,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> Union[str, Generator[str, None, None], Dict[str, int]]:
        """
        Chat interface that supports both streaming and non-streaming responses.
        """
        try:
            start_time = time.time()
            
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=stream,
                max_tokens=max_tokens,
                temperature=temperature if temperature is not None else 0.7,
                top_p=top_p if top_p is not None else 0.95
            )

            end_time = time.time()
            elapsed_time = end_time - start_time

            if stream:
                def stream_generator():
                    for chunk in response:
                        if chunk.choices[0].delta.content is not None:
                            yield chunk.choices[0].delta.content
                return stream_generator()
            else:
                # Extract token usage
                token_usage = response.usage
                total_tokens = token_usage.total_tokens  # includes prompt + response
                prompt_tokens = token_usage.prompt_tokens
                completion_tokens = token_usage.completion_tokens

                # Calculate TPS
                tps = completion_tokens / elapsed_time if elapsed_time > 0 else 0
                
                return {
                    "response": response.choices[0].message.content,
                    "total_tokens": total_tokens,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "elapsed_time": elapsed_time,
                    "tokens_per_second": tps
                }

        except Exception as e:
            raise RuntimeError(f"Chat failed: {str(e)}")
