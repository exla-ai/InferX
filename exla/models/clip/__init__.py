from exla.utils.device_detect import detect_device
from exla.utils.dependency_manager import ensure_dependencies

def clip():
    """
    Factory function that returns the appropriate CLIP model
    based on the detected hardware.
    """
    # First, ensure all required dependencies are installed
    ensure_dependencies('clip', [
        'clip-client',
        'clip-server'
    ])
    
    # Now we can safely import the implementations
    from ._implementations import Clip_CPU, Clip_Orin_Nano, Clip_GPU
    
    device_info = detect_device()
    device_type = device_info['type']
    
    model_map = {
        "orin_nano": Clip_Orin_Nano,
        "gpu": Clip_GPU, 
        "cpu": Clip_CPU,
    }
    
    if device_type not in model_map:
        raise ValueError(f"No Clip implementation for device type: {device_type}")
        
    return model_map[device_type]()

__all__ = ['clip']