from inferx.utils.device_detect import detect_device
from .internvl2_5_mps import InternVL2_5_MPS
# from .internvl2_5_gpu import InternVL2_5_GPU


def internvl2_5(model_size="1B", use_mpo=False):
    """
    Factory function that returns the appropriate InternVL2_5 model
    based on the detected hardware.
    
    Args:
        model_size: Size of the model to use. Options are "1B", "2B", "4B", "8B"
        use_mpo: Whether to use the MPO (Mixed Preference Optimization) version
                of the model, which has better reasoning capabilities.
    """
    device_info = detect_device()
    
    device_type = device_info['type']
    
    model_map = {
        "mps": InternVL2_5_MPS,  
        # "gpu": InternVL2_5_GPU,
    }
    
    if device_type not in model_map:
        raise ValueError(f"No InternVL2_5 implementation for device type: {device_type}")
        
    return model_map[device_type](model_size=model_size, use_mpo=use_mpo)

__all__ = ['internvl2_5']