import io
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch_tensorrt
from trt_eval2 import SAM2FullModel
from pydantic import BaseModel
from typing import List
import uvicorn

app = FastAPI()

# Initialize model and predictor globally
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
encoder = predictor.model.eval().cuda()
sam_model = SAM2FullModel(encoder.half()).eval().cuda()

# Export and compile with TensorRT
def init_trt_model():
    # Sample inputs for tracing
    sample_image = torch.randn(1, 3, 1024, 1024).cuda().half()
    sample_coords = torch.tensor([[500, 375]], dtype=torch.float).cuda().half()
    sample_labels = torch.tensor([1], dtype=torch.int).cuda()
    
    sample_inputs = (sample_image, sample_coords, sample_labels)
    
    # Export the model
    exp_program = torch.export.export(sam_model, sample_inputs, strict=False)
    
    # Compile with TensorRT
    return torch_tensorrt.dynamo.compile(
        exp_program,
        inputs=sample_inputs,
        min_block_size=1,
        enabled_precisions={torch.float16},
        use_fp32_acc=True,
    )

# Initialize TensorRT model
trt_model = init_trt_model()

class PredictionResponse(BaseModel):
    low_res_masks: List[List[List[List[float]]]]
    iou_predictions: List[List[float]]

def preprocess_inputs(image, predictor):
    """Preprocess image and create point prompts."""
    # Resize image to 1800x1200 as expected by the client
    if image.size != (1800, 1200):
        image = image.resize((1800, 1200))
    
    # Convert to numpy array and preprocess
    image_array = np.array(image)
    input_image = predictor._transforms(image_array)[None, ...].to("cuda:0")

    # Fixed point coordinates for now
    point_coords = torch.tensor([[500, 375]], dtype=torch.float).to("cuda:0")
    point_labels = torch.tensor([1], dtype=torch.int).to("cuda:0")

    # Transform coordinates
    orig_hw = (image.size[1], image.size[0])  # (height, width)
    point_coords = torch.as_tensor(point_coords, dtype=torch.float, device=predictor.device)
    unnorm_coords = predictor._transforms.transform_coords(
        point_coords, normalize=True, orig_hw=orig_hw
    )
    labels = torch.as_tensor(point_labels, dtype=torch.int, device=predictor.device)
    
    if len(unnorm_coords.shape) == 2:
        unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]

    # Convert to half precision
    input_image = input_image.half()
    unnorm_coords = unnorm_coords.half()

    return (input_image, unnorm_coords, labels)

def postprocess_masks(out, predictor, image):
    """Postprocess masks and scores."""
    orig_hw = (image.size[1], image.size[0])  # (height, width)
    masks = predictor._transforms.postprocess_masks(out["low_res_masks"], orig_hw)
    masks = (masks > 0.0).squeeze(0).cpu().numpy()
    scores = out["iou_predictions"].squeeze(0).cpu().numpy()
    
    # Sort by scores
    sorted_indices = np.argsort(scores)[::-1]
    masks = masks[sorted_indices]
    scores = scores[sorted_indices]
    
    return masks, scores

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Endpoint for SAM2 inference."""
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Preprocess inputs
        inputs = preprocess_inputs(image, predictor)
        
        # Run inference
        with torch.no_grad():
            output = sam_model(*inputs)
        
        # Convert outputs for JSON serialization
        low_res_masks = output["low_res_masks"].cpu().float().numpy()
        iou_predictions = output["iou_predictions"].cpu().float().numpy()
        
        return PredictionResponse(
            low_res_masks=low_res_masks.tolist(),
            iou_predictions=iou_predictions.tolist()
        )
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
