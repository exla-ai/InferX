class Clip_Base:
    def __init__(self):
        super().__init__()

    def inference(self, image_paths, classes=[]):
        print(f"Running inference on {self.__class__.__name__}")
        return ["test_prediction"]
