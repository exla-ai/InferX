from exla.optimize import optimize_model

model_path = "models/custom_model/model.safetensors"

config = {}


optimized_model = optimize_model(model_path, **config)

