from exla.utils.device_detect import detect_device
from exla.utils.dependency_manager import ensure_dependencies

def deepseek_r1():
    """
    Factory function that returns the appropriate DeepSeek R1 model
    based on the detected hardware.
    """
    device_info = detect_device()
    device_type = device_info['type']
    
    # Base dependencies for DeepSeek
    base_deps = [
        'transformers>=4.34.0',
        'accelerate'
    ]
    
    # Hardware-specific dependencies
    hw_deps = None
    if device_type == 'orin_nano':
        hw_deps = [
            'tensorrt',
            'Jetson.GPIO'
        ]
    elif device_info['capabilities']['cuda_available']:
        cuda_version = device_info.get('cuda_version', '')
        hw_deps = [
            f'torch>=2.0.0+cu{cuda_version}' if cuda_version else 'torch>=2.0.0',
            'vllm'
        ]
    
    # Ensure all dependencies are installed
    ensure_dependencies('deepseek', base_deps, hw_deps)
    
    # Now we can safely import the implementations
    from ._implementations import Deepseek_R1_CPU, Deepseek_R1_GPU
    
    model_map = {
        "gpu": Deepseek_R1_GPU,
        "cpu": Deepseek_R1_CPU,
    }
    
    if device_type not in model_map:
        raise ValueError(f"No DeepSeek R1 implementation for device type: {device_type}")
        
    return model_map[device_type]()

__all__ = ['deepseek_r1']