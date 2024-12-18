from ._base import MobileNetBase
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm

class MobileNetA100(MobileNetBase):
    def __init__(self):
        super().__init__()
        print("Initializing A100 Model")
        # Load MobileNetV2 pretrained model
        self.model = models.mobilenet_v2(pretrained=True)
        
        # Replace classifier for your number of classes
        num_classes = 10  # Adjust based on your dataset
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features, 
            num_classes
        )
        
        self.device = torch.device("cuda")
        self.model.to(self.device)
        
        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
    def train(self, data_loader, num_epochs=10):
        print("Training on A100")
        self.model.train()
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Progress bar for each epoch
            pbar = tqdm(data_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for batch in pbar:
                images = batch['image'].to(self.device)
                # Assuming your dataset provides labels
                labels = batch['label'].to(self.device)
                
                # Zero the gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': running_loss/len(pbar),
                    'acc': 100.*correct/total
                })
        
        print(f"Training completed. Final accuracy: {100.*correct/total:.2f}%")
        return self
        
    def inference(self, input_data):
        print("Running inference on A100")
        self.model.eval()
        
        with torch.no_grad():
            if isinstance(input_data, str):
                # Single image inference
                dataset = ImageDataset(input_data)
                image = dataset[0]['image'].unsqueeze(0).to(self.device)
                output = self.model(image)
                _, predicted = output.max(1)
                return predicted.tolist()
            else:
                # Batch inference
                predictions = []
                for batch in input_data:
                    images = batch['image'].to(self.device)
                    outputs = self.model(images)
                    _, predicted = outputs.max(1)
                    predictions.extend(predicted.tolist())
                return predictions