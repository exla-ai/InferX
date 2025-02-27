#!/bin/bash
# Script to run the RoboPoint demo pipeline

# Set the base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXAMPLES_DIR="$BASE_DIR/examples"
SAMPLE_IMAGES_DIR="$BASE_DIR/sample_images"
OUTPUT_DIR="$BASE_DIR/outputs"

# Print header
echo "====================================="
echo "RoboPoint Demo Pipeline"
echo "====================================="
echo

# Create directories
mkdir -p "$SAMPLE_IMAGES_DIR"
mkdir -p "$OUTPUT_DIR"

# Generate a test image
echo "Step 1: Generating a test image..."
python "$EXAMPLES_DIR/generate_test_image.py" --output_dir "$SAMPLE_IMAGES_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Failed to generate test image."
    exit 1
fi
echo

# Check if the test image was created
if [ ! -f "$SAMPLE_IMAGES_DIR/test_table.jpg" ]; then
    echo "Error: Test image was not generated."
    exit 1
fi

# Set the image path
SAMPLE_IMAGE_PATH="$SAMPLE_IMAGES_DIR/test_table.jpg"

# Run the RoboPoint example
echo "Step 2: Running RoboPoint example with image: $(basename "$SAMPLE_IMAGE_PATH")"
python "$EXAMPLES_DIR/robotpoint_example.py" --image "$SAMPLE_IMAGE_PATH" --output_dir "$OUTPUT_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Failed to run RoboPoint example."
    exit 1
fi
echo

# Print summary
echo "====================================="
echo "Demo completed successfully!"
echo "====================================="
echo "Sample images: $SAMPLE_IMAGES_DIR"
echo "Output visualizations: $OUTPUT_DIR"
echo
echo "To run with a different image, use:"
echo "python $EXAMPLES_DIR/robotpoint_example.py --image <path_to_image> --output_dir $OUTPUT_DIR"
echo 