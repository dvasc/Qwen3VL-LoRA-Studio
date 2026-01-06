import os
import sys
import threading
import time
import logging # <--- ADD THIS IMPORT
from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv

# Cloud Access Layer
try:
    from pyngrok import ngrok, conf
    NGROK_AVAILABLE = True
except ImportError:
    NGROK_AVAILABLE = False

from engine.trainer import LoraTrainer
from engine.monitoring import get_hardware_status

# ============================================================================
# Initialization & Configuration
# ============================================================================

# 1. Load Environment Variables (for Cloud/Ngrok config)
load_dotenv()

app = Flask(__name__)

# 2. Critical Configuration
# Increase max upload size to 2GB for large JSONL datasets
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB

# 3. Directory Setup
UPLOAD_FOLDER = "uploads"
UPLOAD_FOLDER_TRAIN = os.path.join(UPLOAD_FOLDER, "train")
UPLOAD_FOLDER_VAL = os.path.join(UPLOAD_FOLDER, "val")
OUTPUT_FOLDER = "outputs"
TEMP_FOLDER = "temp"

os.makedirs(UPLOAD_FOLDER_TRAIN, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_VAL, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Configure HuggingFace to use local temp folder via HF_HOME
os.environ["HF_HOME"] = os.path.abspath(TEMP_FOLDER)

# ============================================================================
# Global State Management
# ============================================================================

TRAINING_STATE = {
    "status": "IDLE",      # IDLE, UPLOADING, TRAINING, FINISHED, ERROR, INTERRUPTED
    "progress": 0,
    "logs": [],
    "start_time": None,
    "duration": "00:00",
    "etr": "--:--",
    "output_zip": None,
    "error_msg": None,
    "stop_signal": False,
    "val_metrics": None
}

trainer_thread = None

# ============================================================================
# Routes & Logic
# ============================================================================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_files():
    """
    Handles file uploads for both training and validation sets.
    Requires 'type' form field: 'train' or 'val'.
    """
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    dataset_type = request.form.get("type", "train")
    if dataset_type == "val":
        target_folder = UPLOAD_FOLDER_VAL
    else:
        target_folder = UPLOAD_FOLDER_TRAIN

    files = request.files.getlist("files")
    
    # Clean previous uploads in this specific bucket
    for f in os.listdir(target_folder):
        try:
            os.remove(os.path.join(target_folder, f))
        except Exception:
            pass
    
    saved_paths = []
    for file in files:
        if file.filename and (file.filename.endswith(".jsonl") or file.filename.endswith(".json")):
            filepath = os.path.join(target_folder, file.filename)
            file.save(filepath)
            saved_paths.append(file.filename) 
    
    return jsonify({
        "count": len(saved_paths), 
        "paths": saved_paths,
        "type": dataset_type
    })

@app.route("/train", methods=["POST"])
def start_training():
    global trainer_thread
    
    if TRAINING_STATE["status"] == "TRAINING":
        return jsonify({"error": "Training already in progress"}), 400
    
    config = request.json
    
    # Reset state for new run
    TRAINING_STATE.update({
        "status": "TRAINING",
        "progress": 0,
        "logs": ["Initializing training environment..."],
        "start_time": time.time(),
        "duration": "00:00",
        "etr": "--:--",
        "output_zip": None,
        "error_msg": None,
        "stop_signal": False,
        "val_metrics": None
    })
    
    # Extract "use_thinking" flag from frontend
    use_thinking = config.get("use_thinking", False)
    
    trainer = LoraTrainer(
        base_model=config.get("base_model", "huihui-ai/Huihui-Qwen3-VL-2B-Thinking-abliterated"),
        dataset_folder_train=UPLOAD_FOLDER_TRAIN,
        dataset_folder_val=UPLOAD_FOLDER_VAL,
        output_folder=OUTPUT_FOLDER,
        epochs=int(config.get("epochs", 3)),
        batch_size=int(config.get("batch_size", 1)),
        learning_rate=float(config.get("lr", 2e-4)),
        use_thinking=use_thinking,  # Pass flag to trainer
        state_ref=TRAINING_STATE,
        temp_folder=TEMP_FOLDER,
    )
    
    trainer_thread = threading.Thread(target=trainer.run, daemon=False)
    trainer_thread.start()
    
    return jsonify({"status": "started"})

@app.route("/stop", methods=["POST"])
def stop_training():
    if TRAINING_STATE["status"] == "TRAINING":
        TRAINING_STATE["stop_signal"] = True
        return jsonify({"status": "stopping"})
    return jsonify({"status": "ignored", "message": "Not currently training"})

@app.route("/status")
def get_status():
    return jsonify(TRAINING_STATE)

@app.route("/hardware-status")
def hardware_status():
    """
    Proxy route for hardware telemetry.
    Returns:
        JSON object with 'available' (bool) and either metrics or 'error'.
    """
    return jsonify(get_hardware_status())

@app.route("/download")
def download_lora():
    zip_path = TRAINING_STATE.get("output_zip")
    if zip_path and os.path.exists(zip_path):
        return send_file(zip_path, as_attachment=True)
    return jsonify({"error": "File not found."}), 404

@app.route("/reset", methods=["POST"])
def reset():
    TRAINING_STATE.update({
        "status": "IDLE",
        "progress": 0,
        "logs": [],
        "output_zip": None,
        "error_msg": None,
        "etr": "--:--",
        "duration": "00:00",
        "stop_signal": False,
        "val_metrics": None
    })
    
    # Clear both upload folders
    for folder in [UPLOAD_FOLDER_TRAIN, UPLOAD_FOLDER_VAL]:
        for f in os.listdir(folder):
            try:
                os.remove(os.path.join(folder, f))
            except Exception:
                pass
            
    return jsonify({"status": "reset"})

# ============================================================================
# Cloud / Remote Access Logic
# ============================================================================
def init_ngrok(port):
    """
    Initializes a secure ngrok tunnel if NGROK_AUTHTOKEN is present in env.
    """
    token = os.getenv("NGROK_AUTHTOKEN")
    
    if not token:
        print("\n[DEPLOY] No NGROK_AUTHTOKEN found. Running in Localhost mode.")
        return

    if not NGROK_AVAILABLE:
        print("\n[DEPLOY] pyngrok not installed. Skipping remote tunnel.")
        return

    try:
        # Prevent double-tunneling during Flask reload
        if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
            # Config ngrok
            conf.get_default().auth_token = token
            
            # Open tunnel
            public_url = ngrok.connect(port).public_url
            
            print("\n" + "="*60)
            print(f"🚀 PUBLIC CLOUD ACCESS ENABLED")
            print(f"🔗 ACCESS UI HERE: {public_url}")
            print("="*60 + "\n")
            
            # Log to application state
            TRAINING_STATE["logs"].append(f"System: Remote access tunnel enabled at {public_url}")
            
    except Exception as e:
        print(f"\n[DEPLOY] Failed to start ngrok tunnel: {e}")

# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    # --- LOGGING CONFIGURATION ---
    # Silence the default INFO logs from Werkzeug and Pyngrok to reduce terminal noise.
    # We only want to see warnings and errors from these libraries.
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)
    
    pyngrok_logger = logging.getLogger('pyngrok')
    pyngrok_logger.setLevel(logging.WARNING)
    
    # Determine Port
    port_str = os.getenv("PORT", "5001")
    try:
        port = int(port_str)
    except ValueError:
        port = 5001

    # Attempt to start ngrok (if configured)
    init_ngrok(port)

    # Start Flask
    # Note: debug=True helps dev, but in prod consider debug=False. 
    # For this 'Studio' app, debug=True is often useful for the logs.
    app.run(debug=True, port=port, host="0.0.0.0")