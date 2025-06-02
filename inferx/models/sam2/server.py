from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import numpy as np
import cv2
from pathlib import Path
import tempfile
import json
from ._implementations.sam2_jetson import SAM2_Jetson

app = FastAPI(title="InferX SAM2 Server")
model = None

@app.on_event("startup")
async def startup_event():
    global model
    model = SAM2_Jetson()

@app.post("/inference")
async def inference(
    image: UploadFile = File(...),
    prompt: str = None
):
    """
    Endpoint for SAM2 inference
    
    Args:
        image: Input image file
        prompt: Optional JSON string containing points or box prompts
    """
    try:
        # Create a temporary file to save the uploaded image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_img:
            temp_img.write(await image.read())
            temp_img_path = temp_img.name
            
        # Create a temporary file for output
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_out:
            temp_out_path = temp_out.name
            
        # Parse prompt if provided
        prompt_dict = None
        if prompt:
            prompt_dict = json.loads(prompt)
            
        # Run inference
        result = model.inference(
            input=temp_img_path,
            output=temp_out_path,
            prompt=prompt_dict
        )
        
        # Return the result and output image
        return FileResponse(
            temp_out_path,
            media_type="image/jpeg",
            headers={"X-SAM2-Result": json.dumps(result)}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 