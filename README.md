# 🚀 Qwen3VL LoRA Studio

![Python Version](https://img.shields.io/badge/Python-3.10+-blue.svg)![Framework](https://img.shields.io/badge/Framework-Flask-black.svg)![License](https://img.shields.io/badge/License-MIT-green.svg)![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

A self-contained, browser-based interface for fine-tuning Qwen3-VL series multimodal models using Low-Rank Adaptation (LoRA). This studio eliminates the need for complex command-line operations, providing a streamlined workflow from dataset upload to adapter download, complete with real-time monitoring and a one-click cloud deployment solution for Google Compute Engine (GCE).

---

## ✨ Key Features

* **Friendly Web UI:** Drag-and-drop datasets, configure hyperparameters, and monitor training, all from your browser.
* **Real-Time Monitoring:** Live dashboard for loss, progress, Estimated Time Remaining (ETR), and GPU hardware telemetry (VRAM, utilization, temperature).
* **Multimodal Ready:** Natively handles the complex `messages` format with interleaved image (Base64) and text data required for Qwen3-VL.
* **Cloud-First Deployment:** Includes robust scripts to provision and launch the studio on a GCE GPU instance, with secure remote access via `ngrok`.
* **Efficient QLoRA Training:** Utilizes `peft`, `bitsandbytes`, and 4-bit quantization to enable fine-tuning on consumer-grade GPUs (e.g., RTX 3060 12GB) and high-end cloud GPUs (L4/A100).
* **Model Flexibility:** Easily switch between different Qwen3-VL variants, including the recommended "Thinking" model.

---

## 🖥️ Web Interface

*(A screenshot of the application UI showing the configuration panel and the real-time training monitor)*

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/dvasc/Qwen3VL-LoRA-Studio.git
cd Qwen3VL-LoRA-Studio```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

This will install PyTorch with CUDA support and all other required packages.

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app.py
```

### 5. Access the Studio

Open your web browser and navigate to **`http://127.0.0.1:5001`**.

---

## ☁️ Cloud Deployment (Google Compute Engine)

Deploy the studio to a high-performance GPU instance on GCE for long training runs.

### 1. Provision GCE Instance

Create a GPU-enabled VM on GCE. For a complete guide, see:

* [**📘 Cloud Provisioning Guide: GCE GPU Instances**](docs/07_gce_provisioning_guide.md) *(You would create this new file or link to the previously generated response)*

### 2. Run the GCE Provisioner

Log into your new VM via SSH and run the setup script. This installs NVIDIA drivers, CUDA, and other system dependencies.

```bash
# Clone the repo first
git clone https://github.com/dvasc/Qwen3VL-LoRA-Studio.git
cd Qwen3VL-LoRA-Studio

# Make scripts executable and run
chmod +x deploy/setup_gce.sh
./deploy/setup_gce.sh
```

*Note: A reboot (`sudo reboot`) might be required after driver installation.*

### 3. Configure Remote Access

Create a `.env` file from the example and add your ngrok authtoken. This token is required for secure remote access.

```bash
cp .env.example .env
nano .env  # Add your token here
```

### 4. Launch the Studio

This script handles the virtual environment, dependency installation, and starts the Flask server with the ngrok tunnel.

```bash
chmod +x deploy/launch.sh
./deploy/launch.sh
```

The terminal will display a public `ngrok.io` URL. Use this URL to access the studio from any device.

---

## 📂 Project Structure

```
/Qwen3VL-LoRA-Studio
├── .venv/                  # Virtual environment
├── deploy/                 # Cloud deployment scripts (GCE, launch)
│   ├── setup_gce.sh
│   └── launch.sh
├── docs/                   # Comprehensive documentation
├── engine/                 # Core training logic
│   ├── config.py           # LoRA & SFT configurations
│   ├── data.py             # Dataset loading & multimodal collation
│   ├── model.py            # Model & processor loading (QLoRA)
│   ├── monitoring.py       # Real-time callbacks & hardware polling
│   └── trainer.py          # Main training orchestrator
├── models/                 # Cached HuggingFace models
├── outputs/                # Trained LoRA adapters
├── static/                 # CSS & JavaScript for the web UI
├── templates/              # HTML templates
├── uploads/                # Uploaded datasets
├── app.py                  # Flask server entry point & routing
├── requirements.txt        # Python dependencies
├── .env.example            # Template for environment variables
└── README.md               # You are here
```

---

## 📚 Documentation

This project contains extensive documentation covering every aspect of its operation, from quick-start guides to deep architectural dives. Start with the navigation guide to find what you need.

* [**🗺️ 06_navigation_guide.md**](docs/06_navigation_guide.md)
* [**🚀 01_quick_reference.md**](docs/01_quick_reference.md)
* [**🏗️ 04_technical_architecture.md**](docs/04_technical_architecture.md)

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
