from exla.utils.device_detect import detect_device
from .whisper_cpu import Whisper_CPU
from .whisper_jetson import Whisper_Jetson
from .whisper_gpu import Whisper_GPU


def whisper(model_name="base.en"):
    """
    Factory function that returns the appropriate Whisper model
    based on the detected hardware.
    
    Args:
        model_name (str): The name of the Whisper model to use.
            Options include: 'tiny.en', 'base.en', 'small.en', 'medium.en', 'tiny', 'base', 'small', 'medium', 'large'
            
    Returns:
        Whisper_Base: An instance of the appropriate Whisper model implementation.
    """
    try:
        device_info = detect_device()
        device_type = device_info['type']
        
        model_map = {
            "orin_nano": Whisper_Jetson,  # Use Jetson implementation for GPU acceleration
            "agx_orin": Whisper_Jetson,   # Use Jetson implementation for GPU acceleration
            "gpu": Whisper_GPU, 
            "cpu": Whisper_CPU,
        }
        
        if device_type not in model_map:
            print(f"Warning: No specific Whisper implementation for device type: {device_type}")
            print("Falling back to CPU implementation.")
            return Whisper_CPU(model_name=model_name)
            
        try:
            return model_map[device_type](model_name=model_name)
        except Exception as e:
            print(f"Error initializing {model_map[device_type].__name__}: {e}")
            print("Falling back to CPU implementation.")
            try:
                return Whisper_CPU(model_name=model_name)
            except Exception as e2:
                print(f"Error initializing CPU fallback: {e2}")
                # Return a dummy implementation that will provide placeholder results
                from ._base import Whisper_Base
                return Whisper_Base()
    except Exception as e:
        print(f"Error detecting device or initializing model: {e}")
        # Return a dummy implementation that will provide placeholder results
        from ._base import Whisper_Base
        return Whisper_Base()

__all__ = ['whisper'] 