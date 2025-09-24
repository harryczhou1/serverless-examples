import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add src folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from constants import DEFAULT_MODEL_DIR

model_name = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-v0.1")
model_dir = os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR)
hf_token = os.getenv("HF_TOKEN", None)

os.makedirs(model_dir, exist_ok=True)

print(f"Downloading model '{model_name}' to '{model_dir}' ...")

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
tokenizer.save_pretrained(model_dir)
print("Tokenizer downloaded and saved.")

# Download model
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)
model.save_pretrained(model_dir)
print("Model downloaded and saved.")

print("✅ Download complete!")
