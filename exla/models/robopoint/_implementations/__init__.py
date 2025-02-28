from exla.utils.device_detect import detect_device


def robopoint(auto_pull=True):
    """
    Factory function that returns the appropriate RoboPoint model
    based on the detected hardware.
    
    Args:
        auto_pull: Whether to automatically pull the Docker image if using GPU implementation.
        
    Returns:
        An instance of the appropriate RoboPoint implementation.
    """
    device_info = detect_device()
    print(device_info)
    device_type = device_info['type']
    
    if device_type == "orin_nano" or device_type == "agx_orin":
        from .robopoint_jetson import RobopointJetson
        return RobopointJetson()
    elif device_type == "gpu":
        from .robopoint_gpu import RobopointGPU
        model = RobopointGPU(auto_pull=auto_pull)
        # Install dependencies (pull Docker image) if auto_pull is True
        if auto_pull:
            model.install_dependencies()
        return model
    elif device_type == "cpu":
        from .robopoint_cpu import RobopointCPU
        return RobopointCPU()
    else:
        raise ValueError(f"No robopoint implementation for device type: {device_type}")


__all__ = ['robopoint'] 