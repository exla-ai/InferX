import torch

def detect_device():
    """
    Automatically detects the hardware device.
    Returns:
        device_name (str): The detected device type.
    Raises:
        ValueError: If the device is unsupported.
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name().lower()
        if "a100" in device_name:
            return "a100"
        elif "orin" in device_name or "nano" in device_name:
            return "orin_nano"
        else:
            raise ValueError(f"Unsupported GPU detected: {device_name}")
    else:
        return "cpu"  # Default to CPU if no CUDA devices are available


