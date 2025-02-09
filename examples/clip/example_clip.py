from exla.models.clip import clip


clip = clip()

predictions = clip.inference(
    image_paths=["data/dog.png", "data/cat.png"],
    classes=["a dog", "a cat", "a bird"]
)


print(predictions)


# Print predictions
# for i, pred in enumerate(predictions):
#     print(f"\nPredictions for image {i+1}:")
#     for match in pred['matches']:
#         print(f"{match['text']}: {match['scores']['clip_score']['value']:.4f}")
