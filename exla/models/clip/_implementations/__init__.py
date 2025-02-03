from ...device_detect import detect_device
from .clip_cpu import Clip_CPU
from .clip_orin_nano import Clip_Orin_Nano



def resnet34():
    """
    Factory function that returns the appropriate Resnet_34 model
    based on the detected hardware.
    """
    device = detect_device()
    
    model_map = {
        "orin_nano": Clip_Orin_Nano,
        "cpu": Clip_CPU,
    }
    
    if device not in model_map:
        raise ValueError(f"No Clip implementation for device: {device}")
        
    return model_map[device]()

__all__ = ['clip']