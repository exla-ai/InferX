from exla.utils.device_detect import detect_device


def robopoint(auto_pull=True, verbosity="warning", server_port=10001):
    """
    Factory function that returns the appropriate RoboPoint model
    based on the detected hardware.
    
    Args:
        auto_pull: Whether to automatically pull the Docker image if using GPU implementation.
        verbosity: Logging level ('debug', 'info', 'warning', 'error', 'critical')
        server_port: Port number for the docker server (default: 10001)
        
    Returns:
        An instance of the appropriate RoboPoint implementation.
    """
    device_info = detect_device()

    device_type = device_info['type']
    
    if device_type == "orin_nano" or device_type == "agx_orin":
        from .robopoint_jetson import RobopointJetson
        return RobopointJetson()
    elif device_type == "gpu":
        from .robopoint_gpu import RobopointGPU
        model = RobopointGPU(auto_pull=auto_pull, verbosity=verbosity, server_port=server_port)
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