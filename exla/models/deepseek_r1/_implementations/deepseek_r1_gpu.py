import time
import signal
import sys
import logging
import atexit
import requests
from pathlib import Path
import docker
from docker.errors import NotFound, APIError
import threading
import itertools
from ._base import Deepseek_R1_Base

# Set up basic logging
logging.basicConfig(level=logging.WARNING)  # Only show warnings and above
logger = logging.getLogger(__name__)

class ProgressIndicator:
    """
    A simple spinner progress indicator.
    
    Usage:
      with ProgressIndicator("Loading service"):
          # do some work
    """
    def __init__(self, message: str, interval: float = 0.1):
        self.message = message
        self.interval = interval
        self._spinner_thread = None
        self._stop_event = threading.Event()
        self._spinner_cycle = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])

    def _animate(self):
        while not self._stop_event.is_set():
            spinner_char = next(self._spinner_cycle)
            sys.stdout.write(f"\r{spinner_char} {self.message}")
            sys.stdout.flush()
            time.sleep(self.interval)
        sys.stdout.write("\r✓ " + self.message + "\n")
        sys.stdout.flush()

    def start(self):
        self._stop_event.clear()
        self._spinner_thread = threading.Thread(target=self._animate)
        self._spinner_thread.daemon = True
        self._spinner_thread.start()

    def stop(self):
        self._stop_event.set()
        if self._spinner_thread:
            self._spinner_thread.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


class DockerManager:
    """A simple Docker container manager using the Docker SDK."""
    def __init__(self):
        self.client = docker.from_env()
        self.containers = {}  # Keep track of started containers

    def pull_image(self, image_name):
        try:
            self.client.images.pull(image_name)
        except APIError as e:
            raise RuntimeError(f"Failed to pull Docker image: {e}")

    def remove_container(self, name):
        try:
            container = self.client.containers.get(name)
            container.remove(force=True)
            self.containers.pop(name, None)
        except NotFound:
            pass
        except APIError as e:
            raise RuntimeError(f"Failed to remove container: {e}")

    def run_container(self, name, image, command=None, ports=None, environment=None, volumes=None, detach=True, **kwargs):
        self.remove_container(name)
        try:
            container = self.client.containers.run(
                image=image,
                command=command,
                name=name,
                ports=ports,
                environment=environment,
                volumes=volumes,
                detach=detach,
                **kwargs
            )
            self.containers[name] = container
            return container
        except APIError as e:
            raise RuntimeError(f"Failed to start container: {e}")

    def stop_all(self):
        for name, container in list(self.containers.items()):
            try:
                container.stop(timeout=10)
                container.remove()
            except APIError:
                pass
            finally:
                self.containers.pop(name, None)


class Deepseek_R1_GPU(Deepseek_R1_Base):
    """
    Manages the lifecycle of the Deepseek language server and the Open WebUI chat container.
    """
    def __init__(self,
                 model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                 server_port=8000,
                 webui_port=3000):
        self.model_name = model_name
        self.server_port = server_port
        self.webui_port = webui_port
        self.docker_manager = DockerManager()
        self.running = True

        # Register signal and exit handlers for graceful cleanup
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self.cleanup)

    def _signal_handler(self, signum, frame):
        print("\nShutting down gracefully...")
        self.running = False
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        """Ensure all Docker containers are stopped and removed."""
        self.docker_manager.stop_all()

    def start_server(self):
        """Start the Deepseek language server container."""
        with ProgressIndicator("Initializing language server"):
            image = "vllm/vllm-openai:latest"
            self.docker_manager.pull_image(image)

            # Build the command-line arguments for vLLM serve.
            cmd = [
                "--model", self.model_name,
                "--host", "0.0.0.0",
                "--port", f"{self.server_port}",
                "--trust-remote-code",
                "--dtype", "half",
                "--served-model-name", "deepseek-r1",
                "--disable-log-requests",
                "--enable-reasoning",               
                "--reasoning-parser", "deepseek_r1", 
                "--enable-log-stats"               
            ]

            # Environment variables for the server.
            env = {
            }

            volumes = {
                f"{str(Path.home())}/.cache/huggingface": {
                    "bind": "/root/.cache/huggingface",
                    "mode": "rw"
                }
            }

            ports = {"8000/tcp": self.server_port}

            # Define device_requests for GPU support.
            from docker.types import DeviceRequest
            device_requests = [DeviceRequest(count=-1, capabilities=[["gpu"]])]

            # Start the server container with GPU support, environment variables, and our updated command.
            self.docker_manager.run_container(
                name="exla-server",
                image=image,
                command=cmd,
                ports=ports,
                volumes=volumes,
                detach=True,
                device_requests=device_requests,
                shm_size="8g",
                environment=env
            )

    def start_webui(self):
        """Start the Open WebUI container."""
        with ProgressIndicator("Initializing chat interface"):
            image = "ghcr.io/open-webui/open-webui:latest"
            self.docker_manager.pull_image(image)

            env = {
                "OPENAI_API_BASE": f"http://localhost:{self.server_port}/v1",
                "OPENAI_API_KEY": "EMPTY",
                "WEBUI_AUTH": "false",
                "MODEL": "deepseek-r1",
                "OPENAI_MODEL": "deepseek-r1",
                "DEFAULT_MODEL": "deepseek-r1",
                "MODELS": "deepseek-r1",
                "COMPLETION_MODEL": "deepseek-r1",
                "CHAT_MODEL": "deepseek-r1",
                "INIT_COMMANDS": "update-models",
                "AVAILABLE_MODELS": "deepseek-r1",
                "MODEL_LIST": "deepseek-r1",
                "FORCE_MODEL_LIST": "true",
                "DEBUG": "true",
                "ENDPOINTS": '[{"name":"DeepSeek","url":"http://localhost:{self.server_port}/v1","api_key":"EMPTY"}]',
                "ENDPOINTS_TYPE": "openai",
                "DISABLE_AUTH": "true"
            }

            volumes = {
                "open-webui": {
                    "bind": "/app/backend/data",
                    "mode": "rw"
                }
            }

            ports = {"8080/tcp": self.webui_port}

            self.docker_manager.run_container(
                name="open-webui",
                image=image,
                environment=env,
                ports=ports,
                volumes=volumes,
                detach=True,
                network="ai-network",
                restart_policy={"Name": "always"}
            )   

    def run(self):
        """Start both services and keep the process alive."""
        self.start_server()
        self.start_webui()
        print(f"\n✨ Chat interface ready at: http://localhost:{self.webui_port}")
        print("\nPress Ctrl+C to stop...")
        
        while self.running:
            time.sleep(1)
