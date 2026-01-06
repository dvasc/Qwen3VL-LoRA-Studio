#!/bin/bash

# ==============================================================================
# Qwen3VL-LoRA-Studio | Application Launch Script
# ==============================================================================
# This script manages the application runtime environment. It handles:
# 1. Automatic updates via Git (Best Effort).
# 2. Configuration (.env) verification and setup.
# 3. Virtual Environment (.venv) creation and self-healing.
# 4. Dependency management via pip.
# 5. Starting the Flask application server.
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# Visual Formatting
GREEN='\033[1;32m'
RED='\033[1;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "\n${GREEN}[LAUNCH] $1${NC}"
}

warn() {
    echo -e "\n${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "\n${RED}[ERROR] $1${NC}"
    exit 1
}

# ------------------------------------------------------------------------------
# 0. Environment Prep
# ------------------------------------------------------------------------------
# Ensure Python can find the NVIDIA driver shared libraries (libnvidia-ml.so)
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

echo "----------------------------------------------------------------"
echo -e "🚀 Preparing Qwen3VL-LoRA-Studio for Launch..."
echo "----------------------------------------------------------------"

# ------------------------------------------------------------------------------
# 1. Auto-Update (Git Pull)
# ------------------------------------------------------------------------------
if [ -d ".git" ]; then
    log "Checking for updates..."
    # We temporarily disable 'set -e' so a failed pull doesn't stop the app from launching
    set +e
    git pull
    EXIT_CODE=$?
    set -e
    
    if [ $EXIT_CODE -eq 0 ]; then
        log "Application is up to date."
    else
        warn "Failed to update repository (Network issue or Merge conflict)."
        echo "   Proceeding with existing local version..."
    fi
else
    warn "Not a git repository. Skipping update check."
fi

# ------------------------------------------------------------------------------
# 2. Configuration Check (.env)
# ------------------------------------------------------------------------------
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  ACTION REQUIRED: .env file not found.${NC}"
    echo "   Configuration is required for remote cloud access."
    echo ""
    read -p "🔑 Please paste your NGROK_AUTHTOKEN and press Enter: " NGROK_TOKEN
    
    if [ -z "$NGROK_TOKEN" ]; then
        error "Token cannot be empty. Launch aborted."
    fi

    echo "NGROK_AUTHTOKEN=$NGROK_TOKEN" > .env
    echo -e "${GREEN}✅ Success! .env file created.${NC}"
fi

# ------------------------------------------------------------------------------
# 3. Virtual Environment Management (Self-Healing)
# ------------------------------------------------------------------------------
VENV_DIR=".venv"

# Check for corrupt state: Directory exists, but 'activate' script is missing
if [ -d "$VENV_DIR" ] && [ ! -f "$VENV_DIR/bin/activate" ]; then
    warn "Corrupt virtual environment detected (missing bin/activate)."
    log "Removing broken environment and attempting fresh creation..."
    rm -rf "$VENV_DIR"
fi

# Create Virtual Environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    log "Creating Python virtual environment..."
    
    # Attempt creation
    if ! python3 -m venv "$VENV_DIR"; then
        echo ""
        echo -e "${RED}❌ Virtual Environment Creation Failed.${NC}"
        echo "   This usually means the 'python3-venv' or 'python3-full' system package is missing."
        echo "   Please run the following command to fix it:"
        echo "   ${YELLOW}sudo apt-get update && sudo apt-get install -y python3-full${NC}"
        exit 1
    fi
fi

# ------------------------------------------------------------------------------
# 4. Dependency Management
# ------------------------------------------------------------------------------
log "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

log "Checking and installing requirements..."
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    warn "requirements.txt not found! Skipping dependency installation."
fi

# ------------------------------------------------------------------------------
# 5. Application Start
# ------------------------------------------------------------------------------
log "Starting Qwen3VL-LoRA-Studio..."
echo "----------------------------------------------------------------"

# Run the application
python3 app.py