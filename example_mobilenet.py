from exla.models.mobilenet._implementations import mobilenet
from exla.utils.data_loader import data_loader

def main():
    # Initialize data loader with a path to your dataset
    train_data = data_loader(path="path/to/training/data")
    
    # Create model - it will automatically detect the hardware
    # and instantiate the appropriate version (A100, Orin, or CPU)
    model = mobilenet()
    
    # Train the model
    trained_model = model.train(train_data)
    
    # Example inference with a single image
    single_image = "path/to/single/image.jpg"
    result = trained_model.inference(single_image)
    print(f"Single image prediction: {result}")
    
    # Example batch inference
    batch_data = data_loader(path="path/to/test/data")
    batch_results = trained_model.inference(batch_data)
    print(f"Batch predictions: {batch_results}")

if __name__ == "__main__":
    main() 