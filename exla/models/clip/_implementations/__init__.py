from ...device_detect import detect_device
from .clip_cpu import Clip_CPU
from .clip_orin_nano import Clip_Orin_Nano
from .clip_gpu import Clip_GPU


def clip():
    """
    Factory function that returns the appropriate CLIP model
    based on the detected hardware.
    """
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