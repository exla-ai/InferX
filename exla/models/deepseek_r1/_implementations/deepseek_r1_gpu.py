from ._base import Deepseek_R1_Base
from typing import List, Dict, Generator, Optional, Union
import subprocess
import threading
import time
import requests
from openai import OpenAI

class Deepseek_R1_GPU(Deepseek_R1_Base):
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", port=8000):
        print("Initializing Deepseek R1 model for GPU...")
        self.model_name = model_name
        self.port = port
        
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

    def chat(self, 
            messages: List[Dict[str, str]], 
            stream: bool = False,
            max_tokens: int = 2048,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None
        ) -> Union[str, Generator[str, None, None]]:
        """
        Chat interface that supports both streaming and non-streaming responses.
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=stream,
                max_tokens=max_tokens,
                temperature=temperature if temperature is not None else 0.7,
                top_p=top_p if top_p is not None else 0.95
            )

            if stream:
                def stream_generator():
                    for chunk in response:
                        if chunk.choices[0].delta.content is not None:
                            yield chunk.choices[0].delta.content
                return stream_generator()
            else:
                return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"Chat failed: {str(e)}")

    