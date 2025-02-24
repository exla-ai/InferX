from exla.models.sam2 import sam2

# Initialize the SAM2 model
model = sam2()

result_image = model.inference_image(
    "data/truck.jpg",
    "data/truck_output/"
)

# print(result_image)

# result_video = model.inference_video(
#     input="data/f1_trimmed.mp4",
#     output="data/f1_trimmed_output.mp4"
# )

# print(result_video)

