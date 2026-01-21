# Dockerfile for VideoPrism Video Embedding Server with GPU Support
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

# Copy the rest of the application
COPY ./videoprism ./videoprism
COPY server.py server.py

# Install the project itself
RUN uv sync --frozen


COPY minio_client.py minio_client.py

# Set environment variables for optimal GPU usage
# Prevent TensorFlow from allocating all GPU memory at startup
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# JAX memory preallocation (90% leaves room for system/other processes)
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Point XLA to system CUDA
ENV XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"

# Expose port for the API server
EXPOSE 8004

# Default command - run the FastAPI server using the uv-managed environment
CMD ["uv", "run", "python", "server.py"]
