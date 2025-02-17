from exla.optimize import optimize_model
import torch

# Load and optimize the model
model_path = "efficientnet_b0_full.pt"
optimized_model = optimize_model(model_path)

# Run inference with batch size 1
input_tensor = torch.randn(1, 3, 224, 224).cuda()
with torch.no_grad():
    predictions = optimized_model(input_tensor)
    
# Print top 5 predictions
probs = torch.nn.functional.softmax(predictions[0], dim=0)
_, top5_indices = torch.topk(probs, 5)
print("\nTop 5 predictions:")
for idx in top5_indices:
    print(f"Class {idx.item()}: {probs[idx].item()*100:.1f}%")

# Note: EfficientNet is trained on ImageNet classes
# Class indices correspond to ImageNet categories (1000 classes)
# For example: 
# 281 = 'tabby cat'
# 282 = 'tiger cat'
# 283 = 'Persian cat'
# etc.

