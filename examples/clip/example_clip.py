from exla.models.clip import clip


clip = clip()


predictions = clip.inference(
    image_paths=["dog.png", "cat.png"],
    classes=["a dog", "a cat", "a bird"]
)