from ...base_model import BaseModel

class Deepseek_R1_Base(BaseModel):
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        super().__init__()
        
    def train(self, data_loader):
        print(f"Training on {self.__class__.__name__}")
        return self
        
    def inference(self, image_paths, classes=[]):
        print(f"Running inference on {self.__class__.__name__}")
        return ["test_prediction"]
