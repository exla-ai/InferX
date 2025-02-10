from exla.models.clip import clip

model = clip()

# Get matches
matches = model.inference(
    image_paths=["data/dog.png", "data/cat.png"],
    text_queries=["a photo of a dog", "a photo of a cat"]
)

print(matches)

# Example output:
# {
#     "a photo of a dog": [
#         {"image_path": "data/dog.png", "score": 0.9921},
#         {"image_path": "data/cat.png", "score": 0.3456}
#     ],
#     "a photo of a cat": [
#         {"image_path": "data/cat.png", "score": 0.8765},
#         {"image_path": "data/dog.png", "score": 0.2345}
#     ]
# }
