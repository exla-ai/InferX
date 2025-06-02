class Robopoint_Base:
    def __init__(self):
        pass

    def inference(self, image_path, text_instruction=None, output=None):
        """
        Run inference with the RoboPoint model to predict keypoint affordances.
        
        Args:
            image_path (str): Path to input image
            text_instruction (str, optional): Language instruction for the model
            output (str, optional): Path to save the output visualization
            
        Returns:
            dict: Results containing keypoints and their coordinates
        """
        return None
        
    def predict_keypoints(self, image_path, text_instruction):
        """
        Predict keypoints based on the image and text instruction.
        
        Args:
            image_path (str): Path to input image
            text_instruction (str): Language instruction for the model
            
        Returns:
            list: List of keypoint coordinates as (x, y) tuples
        """
        return []
        
    def visualize(self, image_path, keypoints, output_path):
        """
        Visualize the predicted keypoints on the image.
        
        Args:
            image_path (str): Path to input image
            keypoints (list): List of keypoint coordinates as (x, y) tuples
            output_path (str): Path to save the visualization
            
        Returns:
            str: Path to the saved visualization
        """
        return output_path 