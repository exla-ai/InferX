import subprocess
import signal
import os
import threading
import sys
import time

class ModelServer:
    def __init__(self, model, port):
        self.model = model
        self.port = port
        self.process = None
        self.stdout_thread = None
        self.stderr_thread = None
        self._should_stop = False
        
    def _stream_output(self, pipe, prefix=''):
        while not self._should_stop:
            line = pipe.readline()
            if not line:
                break
            print(f"{prefix}{line}", end='', flush=True)
            
    def start(self):
        if self.process is not None:
            # Ensure any existing process is properly cleaned up
            self.stop()
            
        self._should_stop = False
        
        if "Llama-3.1-8B-Instruct" in self.model:
            command = f"../llama.cpp-2/build/bin/llama-server -m ../Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --port {self.port} -ngl 999"
        else:
            command = f"../llama.cpp-2/build/bin/llama-server -m ../Falcon3-1B-Instruct-1.58bit-GGUF/ggml-model-tq2_0.gguf --port {self.port} -ngl 999"
        
        print(f"\nStarting server with command: {command}\n")
        
        # Kill any existing process using the port
        try:
            subprocess.run(f"lsof -ti:{self.port} | xargs kill -9", shell=True)
        except:
            pass
            
        self.process = subprocess.Popen(
            command.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        self.stdout_thread = threading.Thread(target=self._stream_output, args=(self.process.stdout, ''))
        self.stderr_thread = threading.Thread(target=self._stream_output, args=(self.process.stderr, 'ERROR: '))
        
        self.stdout_thread.daemon = True
        self.stderr_thread.daemon = True
        
        self.stdout_thread.start()
        self.stderr_thread.start()
        
        # Give the server a moment to start
        time.sleep(1)
        
        return self
    
    def stop(self):
        if self.process is not None:
            print(f"\nStopping server (PID: {self.process.pid})...")
            self._should_stop = True
            
            # First try SIGTERM
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                # If SIGTERM doesn't work, use SIGKILL
                print("Server didn't terminate gracefully, forcing...")
                self.process.kill()
                self.process.wait()
            
            # Clean up threads
            if self.stdout_thread and self.stdout_thread.is_alive():
                self.stdout_thread.join(timeout=1)
            if self.stderr_thread and self.stderr_thread.is_alive():
                self.stderr_thread.join(timeout=1)
            
            # Ensure the port is freed
            try:
                subprocess.run(f"lsof -ti:{self.port} | xargs kill -9", shell=True)
            except:
                pass
            
            self.process = None
            self.stdout_thread = None
            self.stderr_thread = None
            print("Server stopped")

def server(model, port):
    return ModelServer(model, port) 
