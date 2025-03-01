import os
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

# Example of using point prompts and the iamge
result = model.inference(
    input=str(image_path),
    output=str(image_output_dir),
    prompt={"points": [[900, 600]], "labels": [1]}  # Center point
)

result = model.inference(
    input=str(image_path),
    output=str(image_output_dir),
    prompt={"points": [[900, 600]], "labels": [1]}  # Center point
)


# # Example of using box prompts and video
# result = model.inference(
#     input=str(video_path),
#     output=str(video_output_dir),
#     prompt={"box": [400, 400, 1400, 800]}  # [x1, y1, x2, y2]
# )

# Example of using numpy array as input
def simulate_stream():
    """
    Simulates a stream by loading an image as a numpy array.
    
    In a real application, this would be replaced with:
    - Camera feed frames
    - Video frames
    - Network stream frames
    - Or any other source of image data as numpy arrays
    
    Returns:
        str: Path to temporary file containing the image
        
    Note:
        When using SAM2 with numpy arrays directly, you need to:
        1. Save the numpy array to a temporary file
        2. Pass the file path to model.inference()
        3. Clean up the temporary file when done
    """
    import cv2
    import numpy as np
    import tempfile
    
    # Load the image
    img_path = os.path.join(data_dir, "truck.jpg")
    frame = cv2.imread(img_path)
    
    if frame is None:
        print(f"Error: Could not read image {img_path}")
        return None
    
    # In a real stream, you would:
    # 1. Get new frames in a loop
    # 2. Process each frame
    # 3. Display or save results
    
    # For SAM2, we need to save the numpy array to a temporary file
    # because the model expects a file path or a file-like object
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    temp_path = temp_file.name
    temp_file.close()
    
    # Save the numpy array to the temporary file
    cv2.imwrite(temp_path, frame)
    print(f"Saved numpy array to temporary file: {temp_path}")
    
    # Return the path to the temporary file
    # The caller is responsible for deleting this file when done
    return temp_path



# Get a temporary file path for the numpy array
temp_image_path = simulate_stream()

# Process the image - use a custom output directory and prefix for the numpy example
# result = model.inference(
#     input=temp_image_path,
#     output=str(numpy_output_dir),
#     prompt={"box": [400, 400, 1400, 800]},  # [x1, y1, x2, y2]
# )

# Clean up the temporary file
import os
if temp_image_path and os.path.exists(temp_image_path):
    os.remove(temp_image_path)
    print(f"Removed temporary file: {temp_image_path}")

# Check processing time
if "processing_time" in result:
    print(f"Processing took {result['processing_time']:.3f} seconds")

# Check for errors
if result["status"] == "error":
    print(f"Error: {result['error']}")


