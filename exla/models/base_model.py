class BaseModel:
    def __init__(self):
        self.model = None
    
    def train(self, data_loader):
        raise NotImplementedError
        
    def inference(self, input_data):
        raise NotImplementedError 