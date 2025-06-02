from inferx.utils.device_detect import detect_device
from .mobilenet_a100 import MobileNetA100
from .mobilenet_orin_nano import MobileNetOrinNano
from .mobilenet_cpu import MobileNetCPU

def mobilenet():
    """
    Factory function that returns the appropriate MobileNet model
    based on the detected hardware.
    """
    device = detect_device()
    
    model_map = {
        "a100": MobileNetA100,
        "orin_nano": MobileNetOrinNano,
        "cpu": MobileNetCPU
    }
    
    if device not in model_map:
        raise ValueError(f"No MobileNet implementation for device: {device}")
        
    return model_map[device]()

__all__ = ['mobilenet']