import os
import cv2
import numpy as np
import tempfile
import io
from exla.models.sam2 import sam2

data_dir = "data"
    
# Input files
image_path = os.path.join(data_dir, "truck.jpg")
video_path = os.path.join(data_dir, "f1.mp4")
    
# Output directories
image_output_dir = os.path.join(data_dir, "output_truck")
video_output_dir = os.path.join(data_dir, "output_f1")
numpy_output_dir = os.path.join(data_dir, "output_numpy")
    
# Ensure output directories exist
os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(video_output_dir, exist_ok=True)
os.makedirs(numpy_output_dir, exist_ok=True)
    
model = sam2()

# Example 1: Using point prompts and an image
print("\n=== Example 1: Image Processing with Point Prompts ===")
result = model.inference(
    input=str(image_path),
    output=str(image_output_dir),
    prompt={"points": [[900, 600]], "labels": [1]}  # Center point
)

print(f"Image processing result: {result['status']}")
if "processing_time" in result:
    print(f"Processing took {result['processing_time']:.3f} seconds")

# Example 2: Using box prompts and video
print("\n=== Example 2: Video Processing with Box Prompts ===")
result = model.inference(
    input=str(video_path),
    output=str(video_output_dir),
    prompt={"box": [400, 400, 1400, 800]}  # [x1, y1, x2, y2]
)

print(f"Video processing result: {result['status']}")
if "processing_time" in result:
    print(f"Processing took {result['processing_time']:.3f} seconds")

# Example 3: Using numpy array as input
print("\n=== Example 3: Numpy Array Processing ===")

# Create a wrapper around the SAM2 model to handle numpy array inputs
class SAM2ArrayWrapper:
    def __init__(self, model):
        self.model = model
    
    def inference(self, input_array=None, input_path=None, output=None, prompt=None):
        """
        Run inference with the SAM2 model, accepting either a numpy array or a file path.
        
        Args:
            input_array: Numpy array containing the image (BGR format from OpenCV)
            input_path: Path to input image or video (alternative to input_array)
            output: Optional output path to save results
            prompt: Prompt for the model (points, boxes, etc.)
            
        Returns:
            dict: Results from the segmentation
        """
        if input_array is not None:
            # Handle numpy array input by saving to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Save the numpy array to the temporary file
            cv2.imwrite(temp_path, input_array)
            
            try:
                # Run inference using the temporary file
                result = self.model.inference(
                    input=temp_path,
                    output=output,
                    prompt=prompt
                )
                
                return result
            finally:
                # Always clean up the temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        elif input_path is not None:
            # Handle file path input directly
            return self.model.inference(
                input=input_path,
                output=output,
                prompt=prompt
            )
        else:
            raise ValueError("Either input_array or input_path must be provided")

# Create a wrapped model that can handle numpy arrays
wrapped_model = SAM2ArrayWrapper(model)

def save_masks(masks, original_image, output_dir):
    """
    Save the masks as separate image files and create visualizations.
    
    Args:
        masks: List of boolean masks from SAM2
        original_image: Original image as numpy array
        output_dir: Directory to save the masks
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each mask as a separate binary image
    for i, mask in enumerate(masks):
        # Save binary mask (0 or 255)
        mask_image = mask.astype(np.uint8) * 255
        mask_path = os.path.join(output_dir, f"mask_{i+1}.png")
        cv2.imwrite(mask_path, mask_image)
        print(f"  Saved mask {i+1} to {mask_path}")
        
        # Create and save a visualization of the mask on the original image
        vis_image = original_image.copy()
        # Create a colored overlay for the mask
        color = np.random.randint(0, 255, size=3).tolist()
        colored_mask = np.zeros_like(original_image)
        for c in range(3):
            colored_mask[:, :, c] = np.where(mask, color[c], 0)
        
        # Blend the mask with the original image
        alpha = 0.5  # Transparency factor
        vis_image = cv2.addWeighted(vis_image, 1, colored_mask, alpha, 0)
        
        # Add a border around the mask
        contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_image, contours, -1, color, 2)
        
        # Save the visualization
        vis_path = os.path.join(output_dir, f"mask_{i+1}_visualization.jpg")
        cv2.imwrite(vis_path, vis_image)
        print(f"  Saved visualization to {vis_path}")

def process_numpy_array():
    """
    Demonstrates using a numpy array as input to the SAM2 model.
    
    This function:
    1. Loads an image as a numpy array
    2. Passes the numpy array to the wrapped model's inference method
    3. Optionally saves the output if an output path is provided
    4. Saves and visualizes the mask outputs
    """
    # Load the image as a numpy array
    img_path = os.path.join(data_dir, "truck.jpg")
    image_array = cv2.imread(img_path)
    
    if image_array is None:
        print(f"Error: Could not read image {img_path}")
        return
    
    print(f"Loaded image as numpy array, shape: {image_array.shape}")
    
    # Use a box prompt in the center of the image
    h, w = image_array.shape[:2]
    box_prompt = {"box": [w//4, h//4, w*3//4, h*3//4]}  # [x1, y1, x2, y2]
    
    # Option 1: Process without saving output
    print("\nOption 1: Processing without saving output")
    result = wrapped_model.inference(
        input_array=image_array,  # Pass numpy array directly
        prompt=box_prompt
    )
    
    # Option 2: Process and save output
    output_path = os.path.join(numpy_output_dir, "numpy_output.jpg")
    print(f"\nOption 2: Processing and saving output to {output_path}")
    result_with_output = wrapped_model.inference(
        input_array=image_array,  # Pass numpy array directly
        output=output_path,
        prompt=box_prompt
    )
    
    # Print results
    print("\nResults:")
    if result["status"] == "success":
        print(f"  Success! Found {len(result.get('masks', []))} masks")
        
        # Save the masks from the result
        if "masks" in result:
            masks_dir = os.path.join(numpy_output_dir, "masks")
            print(f"\nSaving masks to {masks_dir}")
            save_masks(result["masks"], image_array, masks_dir)
            
            # Example of how to use the masks for further processing
            print("\nExample of using masks for further processing:")
            for i, mask in enumerate(result["masks"]):
                # Calculate the area of each mask (number of pixels)
                area = np.sum(mask)
                print(f"  Mask {i+1} covers {area} pixels ({area / (h*w) * 100:.2f}% of the image)")
                
                # You could do more processing here, such as:
                # - Calculate the bounding box of the mask
                # - Extract the masked region for further analysis
                # - Apply the mask to other images
                # - etc.
    else:
        print(f"  Error: {result.get('error', 'Unknown error')}")
    
    if os.path.exists(output_path):
        print(f"  Output saved to: {output_path}")

# Run the numpy array processing example
process_numpy_array()

print("\nAll examples completed!")


