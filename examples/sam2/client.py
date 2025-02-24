import io
import requests
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, List
import matplotlib
import cv2
matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting

def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        # Vibrant blue with higher opacity
        color = np.array([0.0, 0.447, 1.0, 0.6])  # Strong blue with 0.6 opacity
    
    h, w = mask.shape
    mask_image = np.zeros((h, w, 4), dtype=np.float32)
    mask_image[mask] = color
    
    if borders:
        # Convert mask to uint8 for contour detection
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a separate image for the contours
        contour_image = np.zeros((h, w, 4), dtype=np.float32)
        # Draw white contours with full opacity
        cv2.drawContours(contour_image, contours, -1, (1, 1, 1, 1), thickness=2)
        
        # Combine the mask and contours
        mask_image = np.clip(mask_image + contour_image, 0, 1)
    
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0], pos_points[:, 1],
        color='lime', marker='*',  # Brighter green color
        s=marker_size, edgecolor='white',
        linewidth=2.0,  # Thicker white edge
        zorder=2  # Ensure points are drawn on top
    )
    ax.scatter(
        neg_points[:, 0], neg_points[:, 1],
        color='red', marker='*',
        s=marker_size, edgecolor='white',
        linewidth=2.0,
        zorder=2
    )

def resize_mask(mask, target_size):
    """Resize a binary mask to the target size."""
    mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
    mask_pil = mask_pil.resize(target_size, Image.NEAREST)
    return np.array(mask_pil) > 127

def visualize_masks(
    image: Image.Image,
    masks: np.ndarray,
    scores: np.ndarray,
    point_coords: torch.Tensor,
    point_labels: torch.Tensor,
    title_prefix: str = "",
    save: bool = True
) -> None:
    """Visualize and save masks overlaid on the original image."""
    print(f"Visualizing {len(masks)} masks with shapes: {masks.shape}")
    print(f"Scores shape: {scores.shape}")
    
    # Convert image to numpy array for processing
    image_np = np.array(image)
    image_h, image_w = image_np.shape[:2]
    
    for i, (mask, score) in enumerate(zip(masks, scores)):
        # Resize mask to match image dimensions
        print(f"Resizing mask from {mask.shape} to ({image_h}, {image_w})")
        resized_mask = resize_mask(mask, (image_w, image_h))
        
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        
        # Create a direct overlay for debugging
        if i == 0:  # For the first mask (highest score)
            # Create a colored mask overlay
            mask_color = np.zeros_like(image_np, dtype=np.uint8)
            mask_color[resized_mask] = [0, 114, 255]  # BGR format for OpenCV
            
            # Create a standalone visualization for debugging
            overlay = image_np.copy()
            cv2.addWeighted(mask_color, 0.6, overlay, 0.4, 0, overlay)
            
            # Save the direct overlay for comparison
            cv2.imwrite(f"{title_prefix}_direct_overlay_{i+1}.png", 
                       cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            print(f"Saved direct overlay to {title_prefix}_direct_overlay_{i+1}.png")
        
        # Standard matplotlib visualization
        show_mask(resized_mask, plt.gca(), random_color=(i > 0))
        show_points(point_coords, point_labels, plt.gca())
        plt.title(f"{title_prefix} Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        output_path = f"{title_prefix}_output_mask_{i + 1}.png"
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()
        print(f"Saved mask {i+1} to {output_path}")

def predict_masks(
    image_path: str,
    server_url: str = "http://localhost:8000/predict"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Send image to server for prediction and process the response.
    
    Args:
        image_path: Path to the input image
        server_url: URL of the prediction server
    
    Returns:
        Tuple of (masks, scores)
    """
    # Read and verify image
    image = Image.open(image_path).convert("RGB")
    if image.size != (1800, 1200):
        print(f"Resizing image from {image.size} to (1800, 1200)")
        image = image.resize((1800, 1200))
    
    # Prepare image for sending
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Send request to server
    files = {'file': ('image.png', img_byte_arr, 'image/png')}
    print("Sending request to server...")
    response = requests.post(server_url, files=files)
    
    if response.status_code != 200:
        raise Exception(f"Error from server: {response.text}")
    
    # Process response
    result = response.json()
    print("Got response from server")
    
    # Extract masks and scores
    masks = np.array(result['low_res_masks'])
    scores = np.array(result['iou_predictions'])
    
    print(f"Raw masks shape: {masks.shape}")
    print(f"Raw scores shape: {scores.shape}")
    
    # Convert masks to boolean
    masks = (masks > 0.0)
    
    # Sort by scores
    sorted_indices = np.argsort(scores[0])[::-1]
    masks = masks[0][sorted_indices]
    scores = scores[0][sorted_indices]
    
    print(f"Processed {len(masks)} masks")
    return masks, scores

if __name__ == "__main__":
    # Example usage
    image_path = "data/truck.jpg"
    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    
    # Get predictions from server
    masks, scores = predict_masks(image_path)
    
    # Visualize results
    point_coords = torch.tensor([[500, 375]])  # Same as server's fixed point
    point_labels = torch.tensor([1])
    
    visualize_masks(
        image,
        masks,
        scores,
        point_coords,
        point_labels,
        title_prefix="Torch-TRT"  # Changed to match documentation
    ) 
