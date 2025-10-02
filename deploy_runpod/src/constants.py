import os
from pathlib import Path

# Default device to use if not overridden by environment
DEFAULT_DEVICE = os.getenv("DEVICE", "cuda")

# Default directory where the Hugging Face model will be downloaded / loaded
DEFAULT_MODEL_DIR = str(Path(os.getenv("MODEL_DIR", "/model")).resolve())

# Default Hugging Face model ID (can be overridden in environment)
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-v0.1")

# Hugging Face access token (optional, only needed for gated/private models)
HF_TOKEN = os.getenv("HF_TOKEN", None)
