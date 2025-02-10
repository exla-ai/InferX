from ._base import Clip_Base

class Clip_CPU(Clip_Base):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Initializes a CLIP model on Orin Nano with TensorRT-like optimizations.
        """
        pass

    def inference(self, image_paths, classes=[]):
        pass
       