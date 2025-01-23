import torch
from ultralytics import YOLO
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

# Step 1: Load YOLO11 Model and Export to ONNX
print("Loading YOLO11 model...")
model = YOLO('yolo11n.pt')  # Load YOLO11 model (you can choose from 'yolo11n', 'yolo11s', etc.)

# Dummy input for the ONNX export (adjust based on model input size)
dummy_input = torch.randn(1, 3, 640, 640).cuda()  # Batch size = 1, Channels = 3, Height = 640, Width = 640

# Export to ONNX format
onnx_model_path = "yolo11.onnx"
print(f"Exporting to {onnx_model_path}...")
torch.onnx.export(model.model, dummy_input, onnx_model_path, verbose=True, input_names=['images'], output_names=['output'])

# Step 2: Convert ONNX Model to TensorRT Engine (FP16 optimization)
print("Converting ONNX model to TensorRT engine with FP16 optimization...")
trtexec_command = f"trtexec --onnx={onnx_model_path} --saveEngine=yolo11_trt_fp16.engine --fp16"
print(f"Running command: {trtexec_command}")
import os
os.system(trtexec_command)

# Step 3: Load TensorRT Engine and Prepare for Inference
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Load the optimized TensorRT engine
engine_path = 'yolo11_trt_fp16.engine'
with open(engine_path, "rb") as f:
    engine_data = f.read()

# Create runtime and deserialize the engine
runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(engine_data)

# Create an execution context
context = engine.create_execution_context()

# Step 4: Allocate Memory for Input and Output
input_shape = (1, 3, 640, 640)  # Input shape for YOLO11 (batch size = 1, channels = 3, height = 640, width = 640)
input_memory = cuda.mem_alloc(np.prod(input_shape) * np.float32().itemsize)
output_memory = cuda.mem_alloc(engine.max_batch_size * engine.get_binding_size(1))

# Create bindings for input and output
bindings = [int(input_memory), int(output_memory)]

# Step 5: Prepare Input Image for Inference
image_path = 'assets/test_image.jpg'  # Path to your input image
img = cv2.imread(image_path)

# Resize and preprocess the image to match YOLO11's input size (640x640)
img_resized = cv2.resize(img, (640, 640))
img_input = np.transpose(img_resized, (2, 0, 1)).astype(np.float32)  # Convert to CHW format
img_input = np.expand_dims(img_input, axis=0)  # Add batch dimension
img_input = np.ascontiguousarray(img_input)

# Step 6: Copy Input Data to GPU
cuda.memcpy_htod(input_memory, img_input)

# Step 7: Run Inference
print("Running inference...")
context.execute_v2(bindings)

# Step 8: Copy Output Data Back to CPU
output_data = np.empty([1, 25200, 85], dtype=np.float32)  # Adjust output shape based on YOLO11 model output
cuda.memcpy_dtoh(output_data, output_memory)

# Step 9: Post-Process the Output (Bounding Boxes, Confidence Scores)
def post_process(output_data, conf_threshold=0.5):
    boxes = output_data[..., :4]  # Bounding box coordinates
    confs = output_data[..., 4:5]  # Confidence scores
    class_probs = output_data[..., 5:]  # Class probabilities

    # Apply thresholding
    mask = confs > conf_threshold
    boxes = boxes[mask]
    confs = confs[mask]
    class_probs = class_probs[mask]

    return boxes, confs, class_probs

# Process the output
boxes, confs, class_probs = post_process(output_data)

# Step 10: Visualize Results (Draw Bounding Boxes on Image)
for box in boxes:
    x1, y1, x2, y2 = box
    cv2.rectangle(img_resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

# Step 11: Show the Resulting Image
cv2.imshow("Detection", img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 12: Save Results (Optional)
cv2.imwrite('output/detected_image.jpg', img_resized)
print("Output image saved as 'output/detected_image.jpg'")

