import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# Which model to pull from Hugging Face Hub
model_name = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-v0.1")
model_dir = "/models/mistral"   # updated to match constants.py
hf_token = os.getenv("HF_TOKEN", None)

os.makedirs(model_dir, exist_ok=True)
print(f"⬇️ Downloading {model_name} to {model_dir}")

# Save config explicitly
config = AutoConfig.from_pretrained(model_name, use_auth_token=hf_token)
config.save_pretrained(model_dir)

# Save tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
tokenizer.save_pretrained(model_dir)
print("✅ Tokenizer saved.")

# Save model
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)
model.save_pretrained(model_dir)
print("✅ Model saved.")

print("🎉 Download complete! Everything is now in /models/mistral")
