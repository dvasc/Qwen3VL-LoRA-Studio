#!/bin/bash

# ============================================================================
# Qwen3VL-LoRA-Studio: Cloud Launch Wrapper
# Handles venv activation, dependency sync, and app startup.
# ============================================================================

set -e

# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

echo "----------------------------------------------------------------"
echo "🛠️  Preparing Qwen3VL-LoRA-Studio for Launch..."
echo "----------------------------------------------------------------"

# 1. Environment Configuration Check
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "⚠️  .env file not found. Creating from .env.example..."
        cp .env.example .env
        echo "🛑 ACTION REQUIRED: Please edit the '.env' file and add your NGROK_AUTHTOKEN."
        echo "   Command: nano .env"
        exit 1
    else
        echo "❌ Critical Error: .env.example is missing. Cannot initialize environment."
        exit 1
    fi
fi

# 2. Virtual Environment Management
if [ ! -d ".venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv .venv
fi

echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# 3. Dependency Synchronization
echo "🔄 Synchronizing dependencies (this may take a moment)..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Launch Application
echo "----------------------------------------------------------------"
echo "🚀 STARTING APPLICATION SERVER"
echo "----------------------------------------------------------------"
echo "Point your browser to the ngrok URL shown below once initialized."
echo "Press CTRL+C to stop the server."
echo "----------------------------------------------------------------"

python app.py