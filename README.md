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

## /Qwen3VL-LoRA-Studio

## ☁️ Cloud Deployment (Google Compute Engine)

Deploy the studio to a high-performance GPU instance on GCE.

### 1. Provision GCE Instance

Create a GPU-enabled VM on GCE. For a complete guide, see:

* [**📘 Cloud Provisioning Guide: GCE GPU Instances**](docs/07_Cloud_Provisioning_Guide.md)

### 2. Run the System Provisioner

Log into your new VM via SSH and run the setup script. This installs NVIDIA drivers, CUDA, and other system dependencies.

```bash
# Clone the repo first
git clone https://github.com/dvasc/Qwen3VL-LoRA-Studio.git
cd Qwen3VL-LoRA-Studio

# Make script executable and run
chmod +x deploy/setup_gce.sh
./deploy/setup_gce.sh

# IMPORTANT: Reboot after the script finishes to load drivers
sudo reboot
```

### 3. Launch the Studio & Configure

After rebooting, SSH back into the VM and navigate to the project directory.

```bash
cd Qwen3VL-LoRA-Studio

# Make the launch script executable
chmod +x deploy/launch.sh

# Run the launcher
./deploy/launch.sh
```

> **Architect's Note:** On the first run, the script will **automatically prompt you** to paste your `NGROK_AUTHTOKEN`. It will create and configure the `.env` file for you. On subsequent runs, it will skip this step.

The terminal will display a public `ngrok.io` URL. Use this URL to access the studio from any device.

---

## 📂 Project Structure

---

## 📚 Documentation

This project contains extensive documentation covering every aspect of its operation, from quick-start guides to deep architectural dives. Start with the navigation guide to find what you need.

* [**🗺️ 06_navigation_guide.md**](docs/06_navigation_guide.md)
* [**🚀 01_quick_reference.md**](docs/01_quick_reference.md)
* [**🏗️ 04_technical_architecture.md**](docs/04_technical_architecture.md)

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
