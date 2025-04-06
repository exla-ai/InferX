import platform
import subprocess
import os
import shutil
import sys

def is_jetson_device():
    """
    Check if the current device is a Jetson device by looking for specific files.
    """
    jetson_files = [
        '/etc/nv_tegra_release',
        '/sys/firmware/devicetree/base/model'
    ]
    return any(os.path.exists(f) for f in jetson_files)

def get_jetson_model():
    """
    Get specific Jetson model by reading device tree model file.
    Returns 'agx_orin' or 'orin_nano' based on the model info.
    """
    model_file = '/sys/firmware/devicetree/base/model'
    if os.path.exists(model_file):
        try:
            with open(model_file, 'r') as f:
                model_info = f.read().lower()
                if 'agx' in model_info:
                    return 'agx_orin'
                elif 'nano' in model_info or 'orin' in model_info:
                    return 'orin_nano'
        except Exception:
            pass
    return None

def has_nvidia_gpu():
    """
    Check if NVIDIA GPU is available using system commands.
    """
    # Check if nvidia-smi is available
    if shutil.which('nvidia-smi') is not None:
        try:
            subprocess.check_output(['nvidia-smi'], stderr=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError:
            pass
    
    # Check for NVIDIA GPU in Linux systems
    if platform.system() == 'Linux':
        try:
            lspci_output = subprocess.check_output(['lspci'], text=True).lower()
            return 'nvidia' in lspci_output
        except subprocess.CalledProcessError:
            pass
    
    return False

def get_gpu_info():
    """
    Get detailed GPU information using nvidia-smi if available.
    Returns:
        dict: GPU information or None if no GPU is available
    """
    if not has_nvidia_gpu():
        return None

    try:
        # Get detailed GPU info
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=gpu_name,memory.total,compute_mode', '--format=csv,noheader'],
            stderr=subprocess.DEVNULL
        ).decode()
        gpu_name = output.strip().split(',')[0].strip().lower()
        
        # Get CUDA version if available
        cuda_version = None
        try:
            nvcc_output = subprocess.check_output(['nvcc', '--version'], stderr=subprocess.DEVNULL).decode()
            for line in nvcc_output.split('\n'):
                if 'release' in line.lower():
                    cuda_version = line.split('V')[-1].split('.')[0]  # Get major version
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        return {
            'name': gpu_name,
            'available': True,
            'cuda_version': cuda_version
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        # GPU exists but nvidia-smi failed
        return {
            'name': 'unknown_nvidia_gpu',
            'available': True,
            'cuda_version': None
        }

def has_mps_support():
    """
    Check if the device has MPS (Metal Performance Shaders) support for Apple Silicon.
    """
    # Check if running on macOS
    if platform.system() != 'Darwin':
        return False
    
    # Check if running on Apple Silicon
    is_arm = platform.machine() == 'arm64'
    
    # Try to import torch and check for MPS availability
    try:
        import torch
        has_mps = torch.backends.mps.is_available()
        return is_arm and has_mps
    except (ImportError, AttributeError):
        # If torch is not installed or doesn't have MPS attribute,
        # just check if we're on Apple Silicon
        return is_arm

def detect_device():
    """
    Automatically detects the hardware device and returns appropriate configuration.
    Returns:
        dict: Device configuration including:
            - type: str (cpu, mps, orin_nano, agx_orin, gpu, etc.)
            - capabilities: dict of device capabilities
            - requirements: list of required packages
    """
    device_info = {
        'type': 'cpu',
        'capabilities': {
            'gpu_available': False,
            'cuda_available': False,
            'mps_available': False,
            'tensorrt_supported': False
        },
        'requirements': ['transformers']  # Base requirements without torch
    }

    # Check for Apple Silicon with MPS support
    if has_mps_support():
        device_info['type'] = 'mps'
        device_info['capabilities']['gpu_available'] = True
        device_info['capabilities']['mps_available'] = True
        device_info['requirements'].append('torch>=2.0.0')
    # Check for NVIDIA GPU
    elif has_nvidia_gpu():
        device_info['capabilities']['gpu_available'] = True
        gpu_info = get_gpu_info()
        
        if gpu_info:
            if gpu_info['cuda_version']:
                device_info['capabilities']['cuda_available'] = True
                device_info['requirements'].append(f'torch>=2.0.0+cu{gpu_info["cuda_version"]}')
            else:
                device_info['requirements'].append('torch')  # CPU version if CUDA version unknown

            # Check for Jetson devices
            if is_jetson_device():
                jetson_model = get_jetson_model()
                if jetson_model:
                    device_info['type'] = jetson_model
                    device_info['capabilities']['tensorrt_supported'] = True
                    device_info['requirements'].extend(['tensorrt', 'cuda-python'])
            # Check GPU name for Jetson devices (fallback)
            elif "orin" in gpu_info['name'].lower():
                if "agx" in gpu_info['name'].lower():
                    device_info['type'] = 'agx_orin'
                else:
                    device_info['type'] = 'orin_nano'
                device_info['capabilities']['tensorrt_supported'] = True
                device_info['requirements'].extend(['tensorrt', 'cuda-python'])
            else:
                device_info['type'] = 'gpu'
                device_info['requirements'].append('cuda-python')
    else:
        device_info['requirements'].append('torch')  # CPU version
    
    return device_info

def get_device_specific_model_path(base_path, device_type):
    """
    Returns the appropriate model path based on device type.
    """
    device_specific_path = os.path.join(base_path, f'model_{device_type}')
    if not os.path.exists(device_specific_path):
        os.makedirs(device_specific_path, exist_ok=True)
    return device_specific_path


