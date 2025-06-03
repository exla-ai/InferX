# SAM2 for Jetson

This module provides an optimized implementation of the Segment Anything Model 2 (SAM2) for NVIDIA Jetson devices.

## Features

- **TensorRT Optimization**: Accelerated inference using NVIDIA TensorRT
- **Camera Stream Support**: Process live video from cameras or RTSP streams
- **Custom Prompts**: Specify points or boxes to guide segmentation

- **Server Mode**: Option to use a remote server for inference

## Installation

The SAM2 model is included in the InferX. Dependencies will be automatically installed when you first use the model.

## Usage (Parts of SAM 2 are broken and need to be fixed like the custom prompts)

### Basic Usage

```python
from inferx.models.sam2 import sam2

# Initialize the model
model = sam2()

# Run inference on an image
result = model.inference(
    input="path/to/image.jpg",
    output="path/to/output.jpg"
)

print(f"Inference result: {result['status']}")
print(f"Found {len(result.get('masks', []))} masks")
```

### Custom Prompts

You can guide the segmentation by providing custom points or boxes:

```python
# Define custom points (x, y coordinates)
custom_points = {
    "points": [[100, 100], [200, 200]],
    "labels": [1, 0]  # 1=foreground, 0=background
}

# Run inference with custom points
result = model.inference(
    input="path/to/image.jpg",
    output="output_custom_points.jpg",
    prompt=custom_points
)
```

### Camera Stream Processing

Process live video from a camera or RTSP stream:

```python
# Process camera stream (camera index 0)
result = model.inference_camera(
    camera_source=0,
    output="camera_output.mp4",
    max_frames=300,  # Process 300 frames
    fps=30,
    display=True  # Show live preview
)

# Process RTSP stream
result = model.inference_camera(
    camera_source="rtsp://example.com/stream",
    output="stream_output.mp4",
    display=True
)
```

### Direct Model Access

For advanced usage, you can get direct access to the underlying model:

```python
# Get direct access to the SAM model
sam_model = model.get_model()

# Get direct access to the predictor
predictor = model.get_predictor()

# Use the predictor directly
image = cv2.imread("image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor.set_image(image_rgb)

# Define custom points
input_point = np.array([[image.shape[1]//2, image.shape[0]//2]])
input_label = np.array([1])

# Get masks
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True
)
```

## Examples

Check the `examples` directory for complete usage examples:

- `sam2_examples.py`: Demonstrates all features of the SAM2 model

Run the examples with:

```bash
python -m inferx.models.sam2.examples.sam2_examples --mode [image|camera|model_access]
```