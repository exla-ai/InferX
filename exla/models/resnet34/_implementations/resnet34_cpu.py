import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
from ._base import Resnet34_Base

class Resnet34_CPU(Resnet34_Base):
    def __init__(self):
        super().__init__()
        # Initialize with pretrained weights but remove the classifier
        self.model = models.resnet34(pretrained=True)
        self.model.fc = None  # Will be set during training based on dataset
        self.device = torch.device('cpu')
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        
    def _initialize_classifier(self, num_classes):
        """Dynamically initialize the classifier based on dataset."""
        if self.model.fc is None or self.model.fc.out_features != num_classes:
            in_features = self.model.fc.in_features if self.model.fc else 512
            self.model.fc = nn.Linear(in_features, num_classes)
            # Reset optimizer with new parameters
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    def train(self, train_loader, val_loader=None, epochs=10, learning_rate=0.001):
        """
        Train the model with automatic class detection and metrics tracking.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
        
        Returns:
            dict: Training history with metrics
        """
        # Detect number of classes from the dataset
        num_classes = len(train_loader.dataset.classes)
        self._initialize_classifier(num_classes)
        
        self.model.to(self.device)
        self.model.train()
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(epochs):
            # Training phase
            running_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': running_loss/len(train_loader),
                    'acc': 100.*correct/total
                })
            
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100. * correct / total
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)
            
            # Validation phase
            if val_loader:
                val_loss, val_acc = self._validate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                print(f'\nEpoch {epoch+1}: '
                      f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            else:
                print(f'\nEpoch {epoch+1}: '
                      f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%')
        
        return history
    
    def _validate(self, val_loader):
        """Run validation and return metrics."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        self.model.train()
        return running_loss / len(val_loader), 100. * correct / total
    
    def inference(self, input_data):
        """
        Run inference on input data.
        
        Args:
            input_data: Preprocessed input tensor or batch of tensors
            
        Returns:
            Predicted class indices
        """
        self.model.eval()
        with torch.no_grad():
            input_data = input_data.to(self.device)
            outputs = self.model(input_data)
            _, predicted = outputs.max(1)
        return predicted.cpu().numpy()
