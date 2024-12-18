from exla.models.mobilenet import mobilenet
from exla.utils.data_loader import data_loader

def main():
    # Initialize data loader with a path to your dataset
    train_data = data_loader(path="examples/mobilenet/sample_data/train")
    
    # Create model - it will automatically detect the hardware
    model = mobilenet()
    
    # Train the model
    trained_model = model.train(train_data)
    
    # Example inference with a single image
    single_image = "examples/mobilenet/sample_data/val/image.jpg"
    result = trained_model.inference(single_image)
    print(f"Single image prediction: {result}")
    
    # Example batch inference
    batch_data = data_loader(path="examples/mobilenet/sample_data/val")
    batch_results = trained_model.inference(batch_data)
    print(f"Batch predictions: {batch_results}")

if __name__ == "__main__":
    main()