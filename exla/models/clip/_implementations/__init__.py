from exla.utils.device_detect import detect_device
from .clip_cpu import Clip_CPU
from .clip_jetson import Clip_Jetson
from .clip_gpu import Clip_GPU


def clip():
    """
    Factory function that returns the appropriate CLIP model
    based on the detected hardware.
    """
    device_info = detect_device()
    device_type = device_info['type']
    
    model_map = {
        "orin_nano": Clip_Jetson,
        "agx_orin": Clip_Jetson,
        "gpu": Clip_GPU, 
        "cpu": Clip_CPU,
    }
    
    if device_type not in model_map:
        raise ValueError(f"No Clip implementation for device type: {device_type}")
        
    return model_map[device_type]()

__all__ = ['clip']