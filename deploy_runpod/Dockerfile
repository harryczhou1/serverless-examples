FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /

ARG MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
ARG TOKENIZER_NAME=mistralai/Mistral-7B-Instruct-v0.2
ARG DEVICE=cuda
ARG MODEL_DIR=/src/model
ARG HF_TOKEN=null

ENV MODEL_NAME=${MODEL_NAME}
ENV TOKENIZER_NAME=${TOKENIZER_NAME}
ENV DEVICE=${DEVICE}
ENV MODEL_DIR=${MODEL_DIR}
ENV HF_TOKEN=${HF_TOKEN}

COPY requirements.txt /requirements.txt
COPY scripts/download_hf_model.py /download_hf_model.py

RUN python3 -m pip install --upgrade -r /requirements.txt --no-cache-dir

# Copy your pre-downloaded model into the container
COPY deploy_runpod/model /src/model

# Copy source code
COPY deploy_runpod/src /src

CMD ["python3", "-u", "/src/handler.py"]
