# Use PyTorch with CUDA runtime
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Work in /app
WORKDIR /app

# Define build args
ARG MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
ARG TOKENIZER_NAME=mistralai/Mistral-7B-Instruct-v0.2
ARG DEVICE=cuda
ARG MODEL_DIR=/model
ARG HF_TOKEN=null

# Set environment variables
ENV MODEL_NAME=${MODEL_NAME}
ENV TOKENIZER_NAME=${TOKENIZER_NAME}
ENV DEVICE=${DEVICE}
ENV MODEL_DIR=/model
ENV HF_TOKEN=${HF_TOKEN}

# Copy requirements and scripts
COPY deploy_runpod/requirements.txt /app/requirements.txt
COPY deploy_runpod/scripts/download_hf_model.py /app/download_hf_model.py

# Install dependencies
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /app/requirements.txt

# (Optional) Download model at build time
# ⚠️ I recommend skipping this for faster builds, let handler.py download at runtime
# RUN python3 /app/download_hf_model.py

# Copy src code
COPY deploy_runpod/src /app/src

# Set RunPod handler
ENV RUNPOD_HANDLER_SRC=src/handler.py

# Default command
CMD ["python3", "-u", "/app/src/handler.py"]
