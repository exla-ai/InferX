import torch
import torch.nn as nn
import torchvision.models as models
import torch.cuda.amp as amp
from ._base import Resnet34_Base

class Resnet34_Orin_Nano(Resnet34_Base):
    def __init__(self):
        super().__init__()
        # Initialize with pretrained weights
        self.model = models.resnet34(pretrained=True)
        self.model.fc = None  # Will be set during training
        
        # Always use CUDA for Orin Nano
        self.device = torch.device('cuda')
        print(f"Using device: {self.device}")
        
        # Initialize loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        
        # Enable TensorRT optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster compute
        torch.backends.cudnn.allow_tf32 = True
        
    def _initialize_classifier(self, num_classes):
        """Initialize the classifier layer with optimized settings."""
        if self.model.fc is None or self.model.fc.out_features != num_classes:
            in_features = self.model.fc.in_features if self.model.fc else 512
            self.model.fc = nn.Linear(in_features, num_classes)
            # Use AdamW with a higher learning rate for faster convergence
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=0.002,
                weight_decay=0.01
            )
            
    def train(self, train_loader, val_loader=None, epochs=10, learning_rate=0.001, num_classes=10):
        """Optimized training loop for Orin Nano."""
        self._initialize_classifier(num_classes)
        
        # Move model to GPU and enable training mode
        self.model = self.model.to(self.device)
        self.model.train()
        
        # Update optimizer with specified learning rate
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Initialize gradient scaler for mixed precision training
        scaler = amp.GradScaler()
        
        # Initialize training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Pre-allocate tensors for metrics to avoid reallocations
        running_loss = torch.zeros(1, device=self.device)
        correct = torch.zeros(1, device=self.device)
        total = torch.zeros(1, device=self.device)
        
        for epoch in range(epochs):
            # Reset metrics
            running_loss.zero_()
            correct.zero_()
            total.zero_()
            
            # Training phase with mixed precision
            self.model.train()
            for inputs, labels in train_loader:
                # Move data to GPU
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Zero gradients
                self.optimizer.zero_grad(set_to_none=True)
                
                # Forward pass with mixed precision
                with amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                
                # Update metrics
                with torch.no_grad():
                    running_loss += loss
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum()
            
            # Calculate epoch metrics
            epoch_loss = (running_loss / len(train_loader)).item()
            epoch_acc = (100. * correct / total).item()
            
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)
            
            # Validation phase
            if val_loader:
                val_loss, val_acc = self._validate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                print(f'Epoch {epoch+1}: '
                      f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            else:
                print(f'Epoch {epoch+1}: '
                      f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%')
        
        return history
    
    def _validate(self, val_loader):
        """Optimized validation loop."""
        self.model.eval()
        running_loss = torch.zeros(1, device=self.device)
        correct = torch.zeros(1, device=self.device)
        total = torch.zeros(1, device=self.device)
        
        with torch.no_grad(), amp.autocast():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum()
        
        val_loss = (running_loss / len(val_loader)).item()
        val_acc = (100. * correct / total).item()
        
        return val_loss, val_acc
    
    def inference(self, input_data):
        """Optimized inference using TensorRT-like optimizations."""
        self.model.eval()
        
        with torch.no_grad(), amp.autocast():
            input_data = input_data.to(self.device, non_blocking=True)
            outputs = self.model(input_data)
            _, predicted = outputs.max(1)
            
        return predicted.cpu().numpy()
