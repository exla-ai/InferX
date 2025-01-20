import subprocess
import signal
import os
import threading
import sys

class ModelServer:
    def __init__(self, model, port):
        self.model = model
        self.port = port
        self.process = None
        
    def _stream_output(self, pipe, prefix=''):
        for line in iter(pipe.readline, ''):
            print(f"{prefix}{line}", end='')
            sys.stdout.flush()
            
    def start(self):
        if self.process is None:
            if "Llama-3.1-8B-Instruct" in self.model:
                command = f"../llama.cpp-2/build/bin/llama-server -m ../Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --port {self.port} -ngl 999"
            else:
                # For optimized model
                command = f"../llama.cpp-2/build/bin/llama-server -m ../Falcon3-1B-Instruct-1.58bit-GGUF/ggml-model-tq2_0.gguf --port {self.port} -ngl 999"
            
            print(f"\nStarting server with command: {command}\n")
            
            self.process = subprocess.Popen(
                command.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True,
                text=True
            )
            
            # Create threads to stream output
            stdout_thread = threading.Thread(target=self._stream_output, args=(self.process.stdout, ''))
            stderr_thread = threading.Thread(target=self._stream_output, args=(self.process.stderr, 'ERROR: '))
            
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            
            stdout_thread.start()
            stderr_thread.start()
            
            return self
    
    def stop(self):
        if self.process is not None:
            print(f"\nStopping server (PID: {self.process.pid})...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)  # Wait up to 5 seconds
            except subprocess.TimeoutExpired:
                print("Server didn't terminate gracefully, forcing...")
                self.process.kill()  # Force kill if it doesn't terminate
                self.process.wait()
            print("Server stopped")
            self.process = None

def server(model, port):
    return ModelServer(model, port) 
