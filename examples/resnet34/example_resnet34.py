import os
import sys
import time
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from exla.models.resnet34 import resnet34

def create_cifar_loaders(num_train_samples=500, batch_size=32):
    """Create CIFAR-10 data loaders with a subset of training data."""
    transform = transforms.Compose([
        transforms.Resize(224),  # ResNet34 expects 224x224 images
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download and load CIFAR-10
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Create a subset of 500 images
    indices = np.random.choice(len(train_dataset), num_train_samples, replace=False)
    train_subset = Subset(train_dataset, indices)
    
    # Create validation set (20% of the subset)
    val_size = int(0.2 * num_train_samples)
    train_indices = indices[:-val_size]
    val_indices = indices[-val_size:]
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, val_loader

def main():
    # Initialize the model
    model = resnet34()
    print(f"Using implementation: {model.__class__.__name__}")
    
    # Create data loaders with 500 training images
    train_loader, val_loader = create_cifar_loaders(num_train_samples=500, batch_size=32)
    print(f"Training on {len(train_loader.dataset)} images")
    print(f"Validating on {len(val_loader.dataset)} images")
    
    # Train the model and get training history
    start_time = time.time()
    history = model.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,
        learning_rate=0.001
    )
    end_time = time.time()
    
    # Print metrics
    training_time = end_time - start_time
    print("\nTraining completed!")
    print(f"Total training time: {training_time:.2f} seconds")
    print(f"Time per image: {training_time/500:.3f} seconds")
    print(f"Final training accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Final validation accuracy: {history['val_acc'][-1]:.2f}%")

if __name__ == "__main__":
    main() 