# Dockerfile for the Video Caption Search Server with GPU Support
# Marlin-2B (captioning) + EmbeddingGemma (text embeddings), PyTorch stack.
# Uses NVIDIA CUDA base image with uv for fast, reproducible dependency management

FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

# Copy uv binary from official image for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set up locale
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install system dependencies including Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Set working directory
WORKDIR /app

# Copy dependency files first for better Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies from lockfile (without installing the project itself yet)
# This layer is cached unless pyproject.toml or uv.lock changes
RUN uv sync --frozen --no-install-project

# Copy the application
COPY captioner.py captioner.py
COPY server.py server.py
COPY storage.py storage.py
COPY minio_client.py minio_client.py
COPY preprocess.py preprocess.py
COPY upsert.py upsert.py
COPY batch_upsert.py batch_upsert.py
COPY client.py client.py

# Install the project itself
RUN uv sync --frozen

# Marlin-2B video preprocessing settings (match the training-time setup)
ENV FORCE_QWENVL_VIDEO_READER=torchcodec
ENV VIDEO_MAX_PIXELS=200704
ENV FPS=2.0
ENV FPS_MAX_FRAMES=240
ENV FPS_MIN_FRAMES=4

# Models are downloaded at runtime on first startup, not at build time.
# They are cached in /root/.cache/huggingface, which is backed by the
# huggingface-cache named volume defined in docker-compose.yml — so the
# download only happens once and persists across container restarts.
#
# EmbeddingGemma is gated: accept the Gemma license on huggingface.co,
# then pass HF_TOKEN via docker-compose.yml (already wired up).

# Expose port for the API server
EXPOSE 8004

# Default command - run the FastAPI server using the uv-managed environment
CMD ["uv", "run", "python", "server.py"]
