from exla.models.clip import clip


clip = clip()

predictions = clip.inference(
    image_paths=["data/dog.png", "data/cat.png"],
    classes=["a dog", "a cat", "a bird"]
)
