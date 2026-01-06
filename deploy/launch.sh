#!/bin/bash

# ============================================================================
# Qwen3VL-LoRA-Studio: Cloud Launch Wrapper (Interactive v2)
# Handles venv activation, dependency sync, interactive .env creation, 
# and application startup.
# ============================================================================

set -e

# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

echo "----------------------------------------------------------------"
echo "🛠️  Preparing Qwen3VL-LoRA-Studio for Launch..."
echo "----------------------------------------------------------------"

# --- Phase 1: Interactive Environment Configuration ---
# Check if the .env file exists. If not, create it interactively.
if [ ! -f ".env" ]; then
    echo "⚠️  ACTION REQUIRED: .env file not found."
    echo "   Configuration is required for remote cloud access."
    echo ""

    # Prompt the user for their token, hiding the input for security (-s flag)
    read -s -p "🔑 Please paste your NGROK_AUTHTOKEN and press Enter: " NGROK_AUTHTOKEN
    echo "" # Adds a newline for cleaner terminal output after the hidden input

    # Validate that the user actually entered something
    if [ -z "$NGROK_AUTHTOKEN" ]; then
        echo "❌ FATAL: NGROK_AUTHTOKEN cannot be empty. Please run the script again."
        exit 1
    fi

    # Create the .env file with the captured token
    echo "NGROK_AUTHTOKEN=$NGROK_AUTHTOKEN" > .env
    # You can add other default variables here if needed in the future
    # echo "PORT=5001" >> .env

    echo "✅ Success! .env file created with your token."
    echo "----------------------------------------------------------------"
fi

# --- Phase 2: Virtual Environment & Dependencies ---
if [ ! -d ".venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv .venv
fi

echo "🔌 Activating virtual environment..."
source .venv/bin/activate

echo "🔄 Synchronizing dependencies (this may take a moment)..."
pip install --upgrade pip
pip install -r requirements.txt

# --- Phase 3: Application Launch ---
echo "----------------------------------------------------------------"
echo "🚀 STARTING APPLICATION SERVER"
echo "----------------------------------------------------------------"
echo "Point your browser to the ngrok URL shown below once initialized."
echo "Press CTRL+C to stop the server."
echo "----------------------------------------------------------------"

python app.py