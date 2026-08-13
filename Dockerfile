FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace/live2d_compiler

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/live2d_compiler

COPY requirements.txt requirements.txt
COPY requirements-vast.txt requirements-vast.txt

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install -r requirements.txt \
    && python3 -m pip install -r requirements-vast.txt \
    && python3 -m pip install git+https://github.com/huggingface/diffusers

RUN git clone --depth 1 https://github.com/xinntao/Real-ESRGAN.git /opt/Real-ESRGAN \
    && cd /opt/Real-ESRGAN \
    && python3 -m pip install -r requirements.txt \
    && python3 setup.py develop \
    && mkdir -p weights \
    && wget -O weights/RealESRGAN_x4plus_anime_6B.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth \
    && python3 - <<'PY'
from pathlib import Path
import torchvision.transforms
site = Path(torchvision.transforms.__file__).parent
(site / "functional_tensor.py").write_text("from torchvision.transforms.functional import *\n", encoding="utf-8")
print("patched torchvision functional_tensor shim")
PY

COPY . .

CMD ["python3", "-m", "src.remote.worker", "--help"]
