import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Which model to pull from Hugging Face Hub
model_name = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-v0.1")
model_dir = "/model"   # fixed location inside container
hf_token = os.getenv("HF_TOKEN", None)

# Ensure /model exists
os.makedirs(model_dir, exist_ok=True)
print(f"⬇️ Downloading {model_name} to {model_dir}")

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
tokenizer.save_pretrained(model_dir)
print("✅ Tokenizer saved.")

# Download model
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)
model.save_pretrained(model_dir)
print("✅ Model saved.")

print("🎉 Download complete! Everything is now in /model")
