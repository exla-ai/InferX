from ._base import Deepseek_R1_Base

class Deepseek_R1_CPU(Deepseek_R1_Base):
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        """
        Initializes a CLIP model on Orin Nano with TensorRT-like optimizations.
        """
        pass

    def train(self):
        pass

    def inference(self, image_paths, classes=[]):
        pass
       