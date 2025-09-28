import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM

# Force import from your repo's src folder so correct constants are used
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from app_constants import DEFAULT_MODEL_DIR  # or constants if you didn’t rename

model_name = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-v0.1")
model_dir = os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR)
hf_token = os.getenv("HF_TOKEN", None)

os.makedirs(model_dir, exist_ok=True)
print(f"Downloading model {model_name} to {model_dir}")

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
tokenizer.save_pretrained(model_dir)
print("Tokenizer saved.")

# Download model
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)
model.save_pretrained(model_dir)
print("Model saved.")

print("✅ Download complete!")
