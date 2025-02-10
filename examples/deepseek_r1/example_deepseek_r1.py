from exla.models.deepseek_r1 import deepseek_r1


model = deepseek_r1()


# Non-streaming example
response = model.chat([
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "What's the capital of France?"}
], stream=False)

print("Non-streaming response:", response)

# Streaming example
print("\nStreaming response:", end=" ")
for chunk in model.chat([
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "What's the capital of France?"}
], stream=True):
    print(chunk, end="", flush=True)
print()