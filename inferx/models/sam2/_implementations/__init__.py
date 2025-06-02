from inferx.utils.device_detect import detect_device
from .sam2_cpu import SAM2_CPU
from .sam2_jetson import SAM2_Jetson
from .sam2_gpu import SAM2_GPU


def sam2():
    """
    Factory function that returns the appropriate SAM2 model
    based on the detected hardware.
    """
    device_info = detect_device()
    device_type = device_info['type']
    
    model_map = {
        "orin_nano": SAM2_Jetson,
        "agx_orin": SAM2_Jetson,
        "gpu": SAM2_GPU, 
        "cpu": SAM2_CPU,
    }
    
    if device_type not in model_map:
        raise ValueError(f"No SAM2 implementation for device type: {device_type}")
        
    return model_map[device_type]()

__all__ = ['sam2'] 