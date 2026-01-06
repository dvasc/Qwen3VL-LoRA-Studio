
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

## Getting Started (Local Development)

### 1. Clone the Repository

```bash
git clone https://github.com/dvasc/Qwen3VL-LoRA-Studio.git
cd Qwen3VL-LoRA-Studio
```

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

Deploy the studio to a high-performance GPU instance on GCE using this streamlined process.

### 1. Initial Setup on VM

Log into your new GCE instance via SSH.

```bash
# 1. Install prerequisites (git and full python environment for venv)
sudo apt-get update && sudo apt-get install -y git python3-full

# 2. Clone the repo and enter the directory
git clone https://github.com/dvasc/Qwen3VL-LoRA-Studio.git
cd Qwen3VL-LoRA-Studio
```

### 2. Run the System Provisioner

This one-click script installs NVIDIA drivers, CUDA, and other system dependencies.

```bash
# Make the script executable and run it
chmod +x deploy/setup_gce.sh
./deploy/setup_gce.sh

# IMPORTANT: Reboot after the script finishes to load drivers
sudo reboot
```

### 3. Launch the Studio

After rebooting, SSH back into the VM and launch the application.

```bash
# Navigate back to the project directory
cd Qwen3VL-LoRA-Studio

# Make the launch script executable and run it
chmod +x deploy/launch.sh
./deploy/launch.sh
```

> On the first run, the script will **prompt you to paste your `NGROK_AUTHTOKEN`**. The terminal will then display a public `ngrok.io` URL. Use this URL to access the studio from any device.

For a more detailed guide on GCE instance creation, see:

* [**📘 Cloud Provisioning Guide: GCE**](docs/Cloud_Provisioning_Guide.md)

---

## 📂 Project Structure

```
/Qwen3VL-LoRA-Studio-app
├── app.py                  # Main Flask application
├── deploy/                 # Deployment scripts for GCE
│   ├── launch.sh           # Starts the application (self-healing)
│   └── setup_gce.sh        # Provisions a GCE instance
├── docs/                   # All project documentation
├── engine/                 # Core training, data, and monitoring logic
│   ├── trainer.py
│   ├── monitoring.py
│   └── ...
├── models/                 # Cached HuggingFace models
├── outputs/                # Saved LoRA adapters
├── static/                 # CSS and JavaScript files
├── templates/              # HTML templates
├── uploads/                # User-uploaded datasets
├── .env.example            # Environment variable template
└── requirements.txt        # Python dependencies
```

---

## 📚 Documentation

This project contains extensive documentation. Start with the navigation guide to find what you need.

* [**🗺️ 06_navigation_guide.md**](docs/06_navigation_guide.md)
* [**🚀 01_quick_reference.md**](docs/01_quick_reference.md)
* [**🏗️ 04_technical_architecture.md**](docs/04_technical_architecture.md)

---

## 📜 License

This project is licensed under the MIT License.
