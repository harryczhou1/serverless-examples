import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from constants import DEFAULT_MODEL_NAME, DEFAULT_MODEL_DIR

model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
model_dir = os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR)

print(f"⬇️ Downloading {model_name} into {model_dir}...")

os.makedirs(model_dir, exist_ok=True)
AutoTokenizer.from_pretrained(model_name).save_pretrained(model_dir)
AutoModelForCausalLM.from_pretrained(model_name).save_pretrained(model_dir)

print("✅ Model saved.")
