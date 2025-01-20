import subprocess
import signal
import os

class ModelServer:
    def __init__(self, model, port):
        self.model = model
        self.port = port
        self.process = None
        
    def start(self):
        if self.process is None:
            if "Llama-3.1-8B-Instruct" in self.model:
                command = f"./llama.cpp-2/build/bin/llama-server -m Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --port {self.port} -ngl 999"
            else:
                # For optimized model
                command = f"./llama.cpp-2/build/bin/llama-server -m Falcon3-1B-Instruct-1.58bit-GGUF/ggml-model-tq2_0.gguf --port {self.port} -ngl 999"
            
            self.process = subprocess.Popen(
                command.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return self
    
    def stop(self):
        if self.process is not None:
            self.process.terminate()
            self.process.wait()
            self.process = None

def server(model, port):
    return ModelServer(model, port) 