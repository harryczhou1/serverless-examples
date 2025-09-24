FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime
WORKDIR /

ARG MODEL_NAME=mistralai/Mistral-7B-v0.1
ARG DEVICE=cuda
ARG MODEL_DIR=/model

ENV MODEL_NAME=${MODEL_NAME}
ENV TOKENIZER_NAME=${MODEL_NAME}
ENV DEVICE=${DEVICE}
ENV MODEL_DIR=${MODEL_DIR}

COPY deploy_runpod/requirements.txt /requirements.txt
COPY deploy_runpod/scripts/download_hf_model.py /download_hf_model.py
COPY deploy_runpod/src /src

RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install --no-cache-dir -r /requirements.txt

# Try downloading model into /model, fallback to HF hub if it fails
RUN python3 /download_hf_model.py || echo "⚠️ Model download failed during build. Will fallback at runtime."

CMD ["python3", "-u", "/src/handler.py"]
