from exla.utils.device_detect import detect_device


def robopoint():
    """
    Factory function that returns the appropriate RoboPoint model
    based on the detected hardware.
    """
    device_info = detect_device()
    print(device_info)
    device_type = device_info['type']
    
    if device_type == "orin_nano" or device_type == "agx_orin":
        from .robopoint_jetson import RoboPointJetson
        return RobopointJetson()
    elif device_type == "gpu":
        from .robopoint_gpu import RobopointGPU
        return RobopointGPU()
    elif device_type == "cpu":
        from .robopoint_cpu import RobopointCPU
        return RobopointCPU()
    else:
        raise ValueError(f"No robopoint implementation for device type: {device_type}")


__all__ = ['robopoint'] 