#!/bin/bash

# ==============================================================================
# Qwen3VL-LoRA-Studio | GCE Provisioning Script
# ==============================================================================
# This script automates the setup of a Google Compute Engine instance for 
# LoRA fine-tuning. It dynamically detects the OS version to install the 
# correct NVIDIA drivers and CUDA toolkit, resolving compatibility issues 
# between Ubuntu 22.04 and 24.04+.
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
    error "Cannot detect OS information. /etc/os-release not found."
fi

log "Detected Environment: OS=$DISTRO, Version=$VERSION_ID ($VERSION_CLEAN), Arch=$ARCH"

# Only support Ubuntu for this specific script logic
if [ "$DISTRO" != "ubuntu" ]; then
    error "This script is optimized for Ubuntu. Detected: $DISTRO"
fi

# ------------------------------------------------------------------------------
# 2. System Updates & Basic Dependencies
# ------------------------------------------------------------------------------
log "Updating system package lists and base dependencies..."
sudo apt-get update
# Non-interactive upgrade to avoid prompts blocking the script
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -yq

# UPDATED: 'python3-full' ensures ensurepip, venv, and distutils are present
# on newer Ubuntu versions (24.04+), preventing virtualenv creation failures.
sudo apt-get install -y wget git python3-pip python3-full build-essential

# ------------------------------------------------------------------------------
# 3. NVIDIA CUDA & Driver Installation (Dynamic)
# ------------------------------------------------------------------------------
log "Configuring NVIDIA CUDA Repositories for ${DISTRO}${VERSION_CLEAN}..."

# Construct the dynamic URL for the NVIDIA keyring
KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}${VERSION_CLEAN}/${ARCH}/cuda-keyring_1.1-1_all.deb"

# Download keyring
wget "$KEYRING_URL" -O cuda-keyring.deb || error "Failed to download NVIDIA keyring from $KEYRING_URL"

# Install keyring
sudo dpkg -i cuda-keyring.deb
sudo apt-get update

log "Installing CUDA Toolkit 12.4 and Drivers..."
# installing 'cuda-drivers' explicitly ensures the kernel modules are present
# installing 'cuda-toolkit-12-4' ensures we have nvcc and libraries
# We use DEBIAN_FRONTEND=noninteractive to suppress keyboard layout prompts etc.
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-12-4 cuda-drivers

# ------------------------------------------------------------------------------
# 4. Environment Configuration
# ------------------------------------------------------------------------------
log "Configuring Path Variables..."

# Add to .bashrc for persistence
if ! grep -q "cuda-12.4" ~/.bashrc; then
    echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
fi

# Export for immediate use in this script session
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

# ------------------------------------------------------------------------------
# 5. Python Application Setup
# ------------------------------------------------------------------------------
log "Installing Python Dependencies..."

# Note: We do not install requirements here anymore. We delegate that to launch.sh
# inside the virtual environment to avoid polluting the system python.

# ------------------------------------------------------------------------------
# 6. Verification
# ------------------------------------------------------------------------------
log "Verifying Installation Status..."

# Check for NVIDIA SMI (Driver Status)
if command -v nvidia-smi &> /dev/null; then
    log "NVIDIA Driver detected:"
    nvidia-smi | head -n 10
else
    error "nvidia-smi not found. Driver installation failed."
fi

# Check for NVCC (Toolkit Status)
if command -v nvcc &> /dev/null; then
    log "CUDA Compiler detected:"
    nvcc --version
else
    error "nvcc not found. CUDA Toolkit installation failed."
fi

log "Setup Completed Successfully! It is HIGHLY RECOMMENDED to reboot the instance now using 'sudo reboot'."