#!/bin/bash
# Environment setup script for PBVS 2026 HISR Challenge
# Run this once on your cloud VM before calling python main.py

set -e
echo "Setting up environment for HISR Challenge..."

# Install PyTorch with CUDA (adjust cuda version as needed)
# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install \
    numpy>=1.24.0 \
    h5py>=3.9.0 \
    scipy>=1.11.0 \
    scikit-image>=0.22.0 \
    requests>=2.31.0 \
    tqdm>=4.65.0 \
    PyYAML>=6.0.1 \
    matplotlib>=3.7.0

echo "Environment setup complete!"
echo "Run: python main.py"
