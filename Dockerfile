FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-pip \
    python3-venv \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/live2d_compiler

COPY requirements.txt requirements.txt
COPY requirements-vast.txt requirements-vast.txt

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install -r requirements.txt \
    && python3 -m pip install -r requirements-vast.txt \
    && python3 -m pip install git+https://github.com/huggingface/diffusers

COPY . .

CMD ["python3", "app.py", "--help"]

