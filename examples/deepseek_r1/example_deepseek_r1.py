from exla.models.deepseek_r1 import deepseek_r1

# Initialize the model - it will automatically select CPU or GPU implementation
model = deepseek_r1()

# Example prompts
prompts = [
    "Write a short poem about artificial intelligence.",
    "Explain quantum computing in simple terms."
]

# Run inference
responses = model.inference(prompts)

# Print results
print("\nGenerated Responses:")
print("-" * 50)
for prompt, response in zip(prompts, responses):
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")
    print("-" * 50)
