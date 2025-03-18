import subprocess
import re
import time

# Define the models to be tested
models = [
    "deepseek-r1:32b",
    "qwen2.5:32b",
    "qwen2.5:7b",
    "mistral-small:24b",
    "phi4:14b",
    "phi3.5:3.8b",
    "llama3.1:8b",
    "llama3.2:3b",
    "qwen2.5:1.5b"
]

# Set the context length and quantization level
context_length = 8000
quantization = "Q4_K_M"

# A more reasonable prompt
prompt_template = """
Please describe the process of photosynthesis in plants. Provide a detailed explanation of the steps involved, including the role of sunlight, carbon dioxide, and water. Also, mention the end - products of photosynthesis and their significance.
"""
# Repeat the prompt to reach the context length
prompt = (prompt_template * (context_length // len(prompt_template)))[:context_length]


# Function to test token rate using ollama --verbose
def test_token_rate(model):
    try:
        start_time = time.time()
        command = [
            "ollama", "run", model, prompt, "--verbose"
        ]
        # Modified part to use Popen
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output = ""
        while True:
            line = process.stdout.readline()
            if not line:
                break
            output += line
        process.wait()
        end_time = time.time()

        # Add a small delay
        time.sleep(0.5)

        #print(output)

        # Extract the prompt eval rate from the verbose output
        prompt_eval_rate_match = re.search(r"prompt eval rate:\s+([\d.]+) tokens/s", output)
        if prompt_eval_rate_match:
            prompt_token_rate = float(prompt_eval_rate_match.group(1))
            print(f"Prompt eval rate for {model}: {prompt_token_rate:.2f} tokens/s")
        else:
            print(f"Could not find prompt eval rate for {model} in the output.")

        # Extract the eval rate from the verbose output
        # Use a more specific pattern to avoid matching prompt eval rate
        eval_rate_match = re.search(r"^(?!prompt )eval rate:\s+([\d.]+) tokens/s", output, re.MULTILINE)
        if eval_rate_match:
            token_rate = float(eval_rate_match.group(1))
            print(f"Eval rate for {model}: {token_rate:.2f} tokens/s")
            return token_rate
        else:
            print(f"Could not find eval rate for {model} in the output.")
            return None
    except Exception as e:
        print(f"Error testing {model}: {e}")
        return None


# Get user input for models to test
print("Available models:")
for idx, model in enumerate(models, start=1):
    print(f"{idx}. {model}")
print("0. All models")

user_choice = input("Enter the number of the model to test (or 0 for all models): ")

if user_choice == '0':
    models_to_test = models
else:
    try:
        choice_idx = int(user_choice) - 1
        if 0 <= choice_idx < len(models):
            models_to_test = [models[choice_idx]]
        else:
            print("Invalid choice. Please try again.")
            models_to_test = []
    except ValueError:
        print("Invalid input. Please enter a valid number.")
        models_to_test = []

# Main testing loop
if models_to_test:
    print("Model\tToken Rate (tokens/second)")
    for model in models_to_test:
        token_rate = test_token_rate(model)
        if token_rate is not None:
            print(f"{model}\t{token_rate:.2f}")