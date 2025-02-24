class SAM2_Base:
    def __init__(self):
        super().__init__()

    def inference(self, input, output=None, prompt=None):
        """
        Run inference with the SAM2 model (generic method).
        
        Args:
            input (str): Path to input image or video
            output (str, optional): Path to save the output
            prompt (dict, optional): Prompt for the model (points, boxes, etc.)
            
        Returns:
            dict: Results from the segmentation
        """
        print(f"Running inference on {self.__class__.__name__}")
        return {"status": "success"}
        
    def inference_image(self, input, output=None, prompt=None):
        """
        Run inference on an image with the SAM2 model.
        
        Args:
            input (str): Path to input image
            output (str, optional): Path to save the output image
            prompt (dict, optional): Prompt for the model (points, boxes, etc.)
            
        Returns:
            dict: Results from the segmentation
        """
        print(f"Running image inference on {self.__class__.__name__}")
        return {"status": "success"}
        
    def inference_video(self, input, output=None, prompt=None):
        """
        Run inference on a video with the SAM2 model.
        
        Args:
            input (str): Path to input video
            output (str, optional): Path to save the output video
            prompt (dict, optional): Prompt for the model (points, boxes, etc.)
            
        Returns:
            dict: Results from the segmentation
        """
        print(f"Running video inference on {self.__class__.__name__}")
        return {"status": "success"} 