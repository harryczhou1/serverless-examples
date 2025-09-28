import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Constants
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
MODEL_DIR = os.path.join(".", "models", "mistral")  # local folder
HF_TOKEN = os.getenv("HF_TOKEN", None)  # optional HuggingFace token

os.makedirs(MODEL_DIR, exist_ok=True)
print(f"Downloading {MODEL_NAME} to {MODEL_DIR}...")

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
tokenizer.save_pretrained(MODEL_DIR)
print("Tokenizer saved.")

# Download model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
model.save_pretrained(MODEL_DIR)
print("Model saved.")

print("✅ Download complete!")
