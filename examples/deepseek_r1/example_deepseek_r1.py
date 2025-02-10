from exla.models.deepseek_r1 import deepseek_r1
import time


model = deepseek_r1()


# Non-streaming example
# response = model.chat([
#     {"role": "system", "content": "You are a helpful AI assistant."},
#     {"role": "user", "content": "What's the capital of France?"}
# ], stream=False)

# print("Non-streaming response:")
# print(f"Response: {response['response']}")
# print(f"Total Tokens: {response['total_tokens']}")
# print(f"Prompt Tokens: {response['prompt_tokens']}")
# print(f"Completion Tokens: {response['completion_tokens']}")
# print(f"Time Elapsed: {response['elapsed_time']} sec")
# print(f"Tokens Per Second (TPS): {response['tokens_per_second']:.2f}")


# Streaming example
print("\nStreaming response:", end=" ")
start_time = time.time()
token_count = 0
output = ""

for chunk in model.chat([
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "What's the capital of France?"}
], stream=True):
    print(chunk, end="", flush=True)
    output += chunk
    token_count += 1  # Approximate token count (each chunk is roughly one token)

end_time = time.time()
elapsed_time = end_time - start_time
tps = token_count / elapsed_time if elapsed_time > 0 else 0

print(f"\nStreaming metrics:")
print(f"Tokens Generated: {token_count}")
print(f"Time Elapsed: {elapsed_time:.2f} sec")
print(f"Tokens Per Second (TPS): {tps:.2f}")