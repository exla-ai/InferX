from exla.utils.device_detect import detect_device
from .robotpoint_cpu import RoboPointCPU
from .robotpoint_gpu import RoboPointGPU
from .robotpoint_jetson import RoboPointJetson


def robotpoint():
    """
    Factory function that returns the appropriate RoboPoint model
    based on the detected hardware.
    """
    device_info = detect_device()
    device_type = device_info['type']
    
    model_map = {
        "orin_nano": RoboPointJetson,  # Use Jetson implementation for GPU acceleration
        "agx_orin": RoboPointJetson,   # Use Jetson implementation for GPU acceleration
        "gpu": RoboPointGPU, 
        "cpu": RoboPointCPU,
    }
    
    if device_type not in model_map:
        raise ValueError(f"No RobotPoint implementation for device type: {device_type}")
        
    return model_map[device_type]()

__all__ = ['robotpoint'] 