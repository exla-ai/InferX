from inferx.utils.device_detect import detect_device
from .optimize_cpu import OptimizeCPU
from .optimize_gpu import OptimizeGPU

def optimize_model(model_path: str, **kwargs):
    """
    Factory function that returns the appropriate optimization strategy
    based on the detected hardware.
    """
    device_info = detect_device()
    device_type = device_info['type']
    
    optimize_map = {
        "gpu": OptimizeGPU,
        "cpu": OptimizeCPU,
    }
    
    if device_type not in optimize_map:
        raise ValueError(f"No optimization implementation for device type: {device_type}")
        
    optimizer = optimize_map[device_type]()
    return optimizer.optimize(model_path, **kwargs)