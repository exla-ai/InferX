from ...device_detect import detect_device
from .deepseek_r1_cpu import Deepseek_R1_CPU
from .deepseek_r1_cpu import Deepseek_R1_GPU


def deepseek_r1():
    """
    Factory function that returns the appropriate Deepseek R1 model
    based on the detected hardware.
    """
    device_info = detect_device()
    device_type = device_info['type']
    
    model_map = {
        "gpu": Deepseek_R1_GPU, 
        "cpu": Deepseek_R1_CPU,
    }
    
    if device_type not in model_map:
        raise ValueError(f"No Deepseek R1 implementation for device type: {device_type}")
        
    return model_map[device_type]()

__all__ = ['deepseek_r1']