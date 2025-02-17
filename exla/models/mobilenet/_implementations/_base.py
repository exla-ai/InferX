class MobileNetBase:
    def __init__(self):
        super().__init__()
        
    def train(self, data_loader):
        print(f"Training on {self.__class__.__name__}")
        return self
        
    def inference(self, input_data):
        print(f"Running inference on {self.__class__.__name__}")
        return ["test_prediction"]
