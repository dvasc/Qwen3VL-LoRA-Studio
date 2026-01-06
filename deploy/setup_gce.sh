#!/bin/bash

# ==============================================================================
# Qwen3VL-LoRA-Studio | GCE Provisioning Script
# ==============================================================================
# This script automates the setup of a Google Compute Engine instance.
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# Function for formatted logging
log() {
    echo -e "\n\033[1;32m[SETUP] $(date +'%Y-%m-%dT%H:%M:%S%z') - $1\033[0m\n"
}

error() {
    echo -e "\n\033[1;31m[ERROR] $(date +'%Y-%m-%dT%H:%M:%S%z') - $1\033[0m\n"
    exit 1
}

log "Initializing Environment Setup..."

# ------------------------------------------------------------------------------
# 1. OS & Architecture Detection
# ------------------------------------------------------------------------------
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO=$ID
    VERSION_CLEAN=$(echo $VERSION_ID | tr -d '.')
    ARCH=$(uname -m)
else
    error "Cannot detect OS information."
fi

log "Detected Environment: OS=$DISTRO, Version=$VERSION_ID, Arch=$ARCH"

# ------------------------------------------------------------------------------
# 2. System Updates & Dependencies
# ------------------------------------------------------------------------------
log "Updating system package lists..."
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -yq

# Install python3-full to ensure venv works on Ubuntu 24.04+
log "Installing Dependencies (including python3-full)..."
sudo apt-get install -y wget git python3-pip python3-full build-essential

# ------------------------------------------------------------------------------
# 3. NVIDIA CUDA & Driver Installation
# ------------------------------------------------------------------------------
log "Configuring NVIDIA CUDA Repositories..."
KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}${VERSION_CLEAN}/${ARCH}/cuda-keyring_1.1-1_all.deb"

wget "$KEYRING_URL" -O cuda-keyring.deb || error "Failed to download NVIDIA keyring"
sudo dpkg -i cuda-keyring.deb
sudo apt-get update

log "Installing CUDA Toolkit 12.4 and Drivers..."
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-12-4 cuda-drivers

# ------------------------------------------------------------------------------
# 4. Environment Configuration
# ------------------------------------------------------------------------------
log "Configuring Path Variables..."

# Add standard Linux library path (for NVML) and CUDA paths
if ! grep -q "cuda-12.4" ~/.bashrc; then
    echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH' >> ~/.bashrc
fi

# Export for immediate use in this session
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# ------------------------------------------------------------------------------
# 5. Verification
# ------------------------------------------------------------------------------
log "Verifying Installation Status..."

if command -v nvidia-smi &> /dev/null; then
    log "NVIDIA Driver detected."
else
    error "nvidia-smi not found. Driver installation failed."
fi

if command -v nvcc &> /dev/null; then
    log "CUDA Compiler detected."
else
    error "nvcc not found. CUDA Toolkit installation failed."
fi

log "Setup Completed Successfully! PLEASE REBOOT NOW using 'sudo reboot'."