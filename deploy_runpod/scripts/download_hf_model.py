import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM

# Compute absolute path to ../src/
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC_DIR)  # force-add it


import constants

model_name = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-v0.1")
model_dir = os.getenv("MODEL_DIR", constants.DEFAULT_MODEL_DIR)
hf_token = os.getenv("HF_TOKEN", None)

os.makedirs(model_dir, exist_ok=True)

print(f"⬇️ Downloading model '{model_name}' to '{model_dir}' ...")

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
tokenizer.save_pretrained(model_dir)
print("✅ Tokenizer downloaded and saved.")

# Download model
model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
model.save_pretrained(model_dir)
print("✅ Model downloaded and saved.")

print("🎉 Download complete!")
