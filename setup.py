import os
import sys
from setuptools import setup
from setuptools.dist import Distribution

def get_device_info():
    """
    Get device information using the existing device detection logic.
    Imports are done inside the function to avoid import errors during initial setup.
    """
    try:
        # Add the current directory to sys.path to import local modules
        sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
        from exla.utils.device_detect import detect_device
        return detect_device()
    except ImportError:
        # If import fails, return CPU configuration
        return {
            'type': 'cpu',
            'capabilities': {
                'gpu_available': False,
                'cuda_available': False,
                'tensorrt_supported': False
            },
            'requirements': ['torch']
        }

class AutoHardwareDistribution(Distribution):
    """Distribution class that automatically detects hardware and sets dependencies"""
    def __init__(self, attrs=None):
        if attrs is None:
            attrs = {}
        
        Distribution.__init__(self, attrs)
        
        # Get core dependencies
        if not attrs.get('install_requires'):
            attrs['install_requires'] = [
                "pillow",
                "psutil",
                "requests",
                "typing-extensions",
                "pydantic"
            ]
        
        # Detect hardware and set appropriate dependencies
        device_info = get_device_info()
        
        # Add hardware-specific dependencies to _cpu, _cuda, or _jetson extras
        if device_info['type'] == 'orin_nano':
            print("📱 Detected Jetson/Orin hardware - enabling Jetson optimizations")
            print("   • Hardware supports TensorRT acceleration")
            # Add Jetson-specific packages to the _jetson extra
            if 'extras_require' not in attrs:
                attrs['extras_require'] = {}
            attrs['extras_require']['_jetson'] = [
                'torch>=2.0.0',
                'Jetson.GPIO',
                'jetson-stats',
                'jtop',
                'tensorrt'
            ]
            # Make _jetson the default hardware platform
            attrs['install_requires'].append('exla-sdk[_jetson]')
        
        elif device_info['capabilities']['cuda_available']:
            cuda_version = device_info.get('cuda_version', '')
            print(f"🚀 Detected CUDA-capable GPU - enabling CUDA optimizations")
            print(f"   • CUDA version: {cuda_version if cuda_version else 'unknown'}")
            
            if 'extras_require' not in attrs:
                attrs['extras_require'] = {}
            
            cuda_deps = ['cuda-python', 'vllm']
            # Add CUDA-specific torch if version is known
            if cuda_version:
                cuda_deps.append(f'torch>=2.0.0+cu{cuda_version}')
            else:
                cuda_deps.append('torch>=2.0.0')
            
            attrs['extras_require']['_cuda'] = cuda_deps
            # Make _cuda the default hardware platform
            attrs['install_requires'].append('exla-sdk[_cuda]')
        
        else:
            print("💻 No specialized hardware detected - using CPU optimizations")
            if 'extras_require' not in attrs:
                attrs['extras_require'] = {}
            attrs['extras_require']['_cpu'] = ['torch']
            # Make _cpu the default hardware platform
            attrs['install_requires'].append('exla-sdk[_cpu]')

        # Add model-specific extras
        if 'extras_require' not in attrs:
            attrs['extras_require'] = {}
        
        attrs['extras_require'].update({
            'clip': [
                'clip-client',
                'clip-server'
            ],
            'deepseek': [
                'transformers>=4.34.0',
                'accelerate'
            ],
            'resnet': [
                'torchvision',
                'tqdm'
            ],
            'mobilenet': [
                'torchvision'
            ],
            'all-models': ['exla-sdk[clip,deepseek,resnet,mobilenet]']
        })

setup(
    distclass=AutoHardwareDistribution,
) 