import logging
import torch
from transformers import TrainerCallback

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HardwareMonitor")

# Safety Check for NVML
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.warning("pynvml not installed. Falling back to PyTorch/Safe Mode.")

# ==============================================================================
# HARDWARE TELEMETRY LOGIC
# ==============================================================================

class HardwareMonitor:
    def __init__(self):
        self.nvml_initialized = False
        
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                logger.info("✅ NVML Initialized Successfully (Advanced Monitoring Active)")
            except Exception as e:
                logger.error(f"⚠️ NVML Initialization Failed: {e}")

    def get_status(self):
        """
        Retrieves hardware telemetry and formats it for the frontend.
        """
        # 1. Initialize Default "Safe" Values
        # We populate every conceivable key name to ensure the Frontend finds what it needs.
        data = {
            # Identification
            "gpu_name": "Scanning...",
            "device_name": "Scanning...",
            "name": "Scanning...",
            "gpu_device": "Scanning...", # Matches UI Label "GPU DEVICE"

            # Load (0-100)
            "load": 0,
            "gpu_load": 0,
            "core_load": 0,          # Matches UI Label "CORE LOAD"
            "utilization": 0,

            # Memory (Bytes) - Most likely what the UI math expects
            "memory_used": 0,
            "memory_total": 0,
            "vram_used": 0,
            "vram_total": 0,
            
            # Memory (GB) - Fallback if UI doesn't do math
            "memory_used_gb": 0,
            "memory_total_gb": 0,
            "vram_used_gb": 0,
            "vram_total_gb": 0,
            "vram_allocation": 0,    # Matches UI Label "VRAM ALLOCATION"

            # Temperature
            "temperature": "--",
            "temp": "--",
            "thermal": "--",         # Matches UI Label "THERMAL"
            "gpu_temp": "--"
        }

        # 2. Strategy A: NVML (Preferred)
        if self.nvml_initialized:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                # Name
                name_bytes = pynvml.nvmlDeviceGetName(handle)
                name = name_bytes.decode("utf-8") if isinstance(name_bytes, bytes) else name_bytes
                
                # Load (%)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                
                # Memory (Bytes)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                raw_used = mem.used
                raw_total = mem.total
                
                # Temp (C)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

                # Update Dictionary
                data.update({
                    "gpu_name": name, "device_name": name, "name": name, "gpu_device": name,
                    "load": util, "gpu_load": util, "core_load": util, "utilization": util,
                    
                    "memory_used": raw_used, "vram_used": raw_used,
                    "memory_total": raw_total, "vram_total": raw_total,
                    
                    "memory_used_gb": round(raw_used / (1024**3), 1),
                    "memory_total_gb": round(raw_total / (1024**3), 1),
                    
                    # Heuristic: If UI label is "VRAM ALLOCATION", it might expect a formatted string or raw bytes
                    "vram_allocation": raw_used, 
                    
                    "temperature": temp, "temp": temp, "thermal": temp, "gpu_temp": temp
                })
            except Exception:
                pass

        # 3. Strategy B: PyTorch Fallback
        elif torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
                raw_used = torch.cuda.memory_allocated(0)
                raw_total = torch.cuda.get_device_properties(0).total_memory
                
                data.update({
                    "gpu_name": name, "device_name": name, "name": name, "gpu_device": name,
                    "memory_used": raw_used, "vram_used": raw_used,
                    "memory_total": raw_total, "vram_total": raw_total,
                    "vram_allocation": raw_used,
                    "memory_used_gb": round(raw_used / (1024**3), 1),
                    "memory_total_gb": round(raw_total / (1024**3), 1)
                })
            except Exception:
                pass

        # DEBUG: Print exactly what we are sending to the UI
        # This will show up in your terminal. If these numbers are > 0, the backend is fixed.
        # logger.info(f"[HardwareMonitor] Payload: {data}")
        
        return data

    def shutdown(self):
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

# ==============================================================================
# CALLBACKS
# ==============================================================================

class EnhancedStateCallback(TrainerCallback):
    def __init__(self, app_state, total_steps):
        self.app_state = app_state
        self.total_steps = total_steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                self.app_state["current_loss"] = round(logs["loss"], 4)
            if self.total_steps > 0:
                progress = (state.global_step / self.total_steps) * 100
                self.app_state["progress"] = round(progress, 1)

# ==============================================================================
# MODULE INTERFACE
# ==============================================================================

_monitor_instance = HardwareMonitor()

def get_hardware_status():
    return _monitor_instance.get_status()