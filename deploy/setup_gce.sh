#!/bin/bash

# ============================================================================
# Qwen3VL-LoRA-Studio: GCE GPU Environment Provisioner
# Target OS: Ubuntu 22.04 LTS
# ============================================================================

set -e  # Exit on error

echo "----------------------------------------------------------------"
echo "🚀 Starting System Provisioning for Qwen3VL-LoRA-Studio"
echo "----------------------------------------------------------------"

# 1. System Updates & Essential Tools
echo "[1/5] Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install -y \
    build-essential \
    software-properties-common \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    curl \
    wget \
    ffmpeg \
    libsm6 \
    libxext6

# 2. NVIDIA Driver Check & Installation
echo "[2/5] Checking for NVIDIA GPU hardware..."
if lspci | grep -i nvidia > /dev/null; then
    echo "✅ NVIDIA GPU detected."
    if ! command -v nvidia-smi &> /dev/null; then
        echo "⏳ Installing NVIDIA Drivers (Headless Server)..."
        sudo add-apt-repository -y ppa:graphics-drivers/ppa
        sudo apt-get update -y
        # Installing version 535 as a stable baseline for CUDA 12.x
        sudo apt-get install -y nvidia-driver-535-server
        echo "⚠️  Drivers installed. A system reboot may be required later."
    else
        echo "✅ NVIDIA Drivers already present."
        nvidia-smi -L
    fi
else
    echo "❌ No NVIDIA GPU detected. Training will be extremely slow on CPU."
fi

# 3. CUDA Toolkit Installation (Targeting CUDA 12.4)
echo "[3/5] Checking for CUDA Toolkit..."
if ! command -v nvcc &> /dev/null; then
    echo "⏳ Installing CUDA Toolkit 12.4..."
    # Download official NVIDIA keyring
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update -y
    sudo apt-get -y install cuda-toolkit-12-4
    
    # Update PATH (for current shell)
    export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    
    # Add to .bashrc for persistence
    echo 'export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
else
    echo "✅ CUDA Toolkit already present."
    nvcc --version
fi

# 4. Git Configuration (Optional)
echo "[4/5] Setting up project directory..."
# Ensuring we are in the project root relative to this script
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# 5. Summary
echo "----------------------------------------------------------------"
echo "✅ PROVISIONING COMPLETE"
echo "----------------------------------------------------------------"
echo "Next Steps:"
echo "1. Reboot if you just installed NVIDIA drivers: 'sudo reboot'"
echo "2. Run the cloud launcher: './deploy/launch.sh'"
echo "----------------------------------------------------------------"