FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    h5py>=3.9.0 \
    scipy>=1.11.0 \
    scikit-image>=0.22.0 \
    requests>=2.31.0 \
    tqdm>=4.65.0 \
    PyYAML>=6.0.1 \
    matplotlib>=3.7.0

# Copy project files
COPY . /workspace/

# Create required directories
RUN mkdir -p data/raw data/train data/val data/test checkpoints predictions

# Default command
CMD ["python", "main.py"]
