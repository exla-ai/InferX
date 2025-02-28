"""
RoboPoint model for keypoint detection in images.
"""

import os
import sys
import platform

# Fix NumPy import issues first
try:
    import numpy
    print(f"NumPy version: {numpy.__version__}")
    # Ensure ndarray is available
    if not hasattr(numpy, 'ndarray'):
        print("NumPy module doesn't have ndarray attribute, reinstalling...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.24.3", "--force-reinstall"])
        import importlib
        importlib.reload(numpy)
        print(f"NumPy reloaded, version: {numpy.__version__}")
except Exception as e:
    print(f"Error with NumPy: {e}")
    print("Installing NumPy...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.24.3", "--force-reinstall"])
    import numpy
    print(f"NumPy installed, version: {numpy.__version__}")

import torch

from ._implementations.robopoint_gpu import RoboPointGPU

def RoboPoint(**kwargs):
    """
    Factory function to create a RoboPoint model instance.
    
    Args:
        **kwargs: Additional arguments to pass to the model.
        
    Returns:
        A RoboPoint model instance.
    """
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available, using GPU implementation")
        return RoboPointGPU(**kwargs)
    else:
        print("CUDA is not available, using CPU implementation")
        # For now, we'll use the GPU implementation but it will run on CPU
        return RoboPointGPU(device="cpu", **kwargs)

# Alias for backward compatibility
robopoint = RoboPoint

__all__ = ['RoboPoint', 'robopoint'] 