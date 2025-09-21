FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime
WORKDIR /

ARG MODEL_NAME=mistralai/Mistral-7B-v0.1
ARG TOKENIZER_NAME=mistralai/Mistral-7B-v0.1
ARG DEVICE=cuda
ARG MODEL_DIR=/model
ARG HF_TOKEN=null

ENV MODEL_NAME=${MODEL_NAME}
ENV TOKENIZER_NAME=${TOKENIZER_NAME}
ENV DEVICE=${DEVICE}
ENV MODEL_DIR=${MODEL_DIR}
ENV HF_TOKEN=${HF_TOKEN}

COPY deploy_runpod/requirements.txt /requirements.txt
COPY deploy_runpod/scripts/download_hf_model.py /download_hf_model.py
COPY deploy_runpod/src /src

# Install dependencies first
RUN python3 -m pip install --upgrade -r /requirements.txt --no-cache-dir

# Try to download model, but don’t fail the build if it errors
RUN python3 /download_hf_model.py || echo "⚠️ Model download failed during build, will retry at runtime."

CMD ["python3", "-u", "/src/handler.py"]
