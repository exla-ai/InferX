from exla.utils.device_detect import detect_device
from .resnet34_orin_nano import Resnet34_Orin_Nano
from .resnet34_cpu import Resnet34_CPU



def resnet34():
    """
    Factory function that returns the appropriate Resnet_34 model
    based on the detected hardware.
    """
    device = detect_device()
    
    model_map = {
        "orin_nano": Resnet34_Orin_Nano,
        "cpu": Resnet34_CPU,
    }
    
    if device not in model_map:
        raise ValueError(f"No Resnet34 implementation for device: {device}")
        
    return model_map[device]()

__all__ = ['resnet34']