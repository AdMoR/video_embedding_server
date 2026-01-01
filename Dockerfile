# Dockerfile for VideoPrism Video Embedding Server with GPU Support
# Uses NVIDIA's prebuilt JAX image with CUDA support for optimal compatibility

# Option 1: Use NVIDIA NGC JAX image (recommended - includes JAX, Flax, CUDA properly configured)
FROM nvcr.io/nvidia/jax:24.10-py3

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set up locale
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install additional system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml setup.py ./

# The NVIDIA JAX image already includes JAX and Flax with proper CUDA support
# Upgrade Flax to the required version if needed
RUN pip install --no-cache-dir --upgrade "flax>=0.11.2"

# Install TensorFlow with CUDA support
# Note: TensorFlow bundles its own CUDA libraries via [and-cuda]
# This may coexist with JAX's CUDA but ensure TF doesn't hog GPU memory
RUN pip install --no-cache-dir "tensorflow[and-cuda]>=2.20.0"

# Install remaining dependencies
RUN pip install --no-cache-dir \
    "numpy>=1.26" \
    "absl-py>=2.3.1" \
    "einops>=0.8.1" \
    "einshape>=1.0" \
    "huggingface-hub>=0.36.0" \
    "mediapy>=1.2.5" \
    "sentencepiece>=0.2.1" \
    pillow

# Copy the rest of the application
COPY . .

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Set environment variables for optimal GPU usage
# Prevent TensorFlow from allocating all GPU memory at startup
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Disable TensorFlow GPU visibility (let JAX use the GPU primarily)
# Uncomment the following line if you want JAX to exclusively use GPU:
# ENV CUDA_VISIBLE_DEVICES_FOR_TF=""

# JAX memory preallocation (90% leaves room for system/other processes)
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Verify GPU setup on container start (optional, comment out for production)
# RUN python -c "import jax; print('JAX devices:', jax.devices())"

# Default command
CMD ["python", "main.py"]
