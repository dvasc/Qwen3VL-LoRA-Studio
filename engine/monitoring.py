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
    logger.warning("pynvml not installed. Falling back to PyTorch for monitoring.")

# ==============================================================================
# HARDWARE TELEMETRY LOGIC
# ==============================================================================

class HardwareMonitor:
    def __init__(self):
        self.nvml_initialized = False
        self.driver_error = False
        
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                logger.info("✅ NVML Initialized Successfully (Advanced Monitoring Active)")
            except Exception as e:
                logger.error(f"⚠️ NVML Initialization Failed: {e}")
                self.driver_error = True

    def get_status(self):
        """
        Retrieves hardware telemetry.
        Returns a 'Key Spam' dictionary containing Raw Bytes, GBs, and various
        naming conventions to ensure the frontend finds what it needs.
        """
        # Default / Scanning State
        data = {
            "name": "Scanning...",
            "gpu_name": "Scanning...",
            "device": "Scanning...",
            
            # Load: Provide both 0-100 integer and formatted string
            "load": 0,
            "gpu_load": 0,
            "utilization": 0,
            "core_load": 0,
            
            # Memory: CRITICAL FIX - Provide RAW BYTES for frontend math
            "memory_used": 0,      # Raw Bytes
            "memory_total": 0,     # Raw Bytes
            "vram_used": 0,
            "vram_total": 0,
            
            # Memory: Provide Pre-calculated GB just in case
            "memory_used_gb": 0,
            "memory_total_gb": 0,
            "vram_used_gb": 0,
            "vram_total_gb": 0,
            
            # Temperature
            "temperature": "--",
            "temp": "--",
            "thermal": "--",
            "gpu_temp": "--"
        }

        # ----------------------------------------------------------------------
        # STRATEGY 1: Advanced NVML Monitoring (Primary)
        # ----------------------------------------------------------------------
        success_nvml = False
        if self.nvml_initialized:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                # Name
                name_bytes = pynvml.nvmlDeviceGetName(handle)
                name = name_bytes.decode("utf-8") if isinstance(name_bytes, bytes) else name_bytes
                
                # Load (Returns % as int, e.g. 45)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                
                # Memory (Returns Bytes)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                raw_used = mem.used
                raw_total = mem.total
                gb_used = round(raw_used / (1024 ** 3), 1)
                gb_total = round(raw_total / (1024 ** 3), 1)
                
                # Temp (Returns C as int, e.g. 65)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

                # Populate ALL variations
                data.update({
                    "name": name, "gpu_name": name, "device": name,
                    
                    "load": util, "gpu_load": util, "utilization": util, "core_load": util,
                    
                    # Raw Bytes (likely what the UI expects for "0 / 0 GB" math)
                    "memory_used": raw_used, "vram_used": raw_used, "mem_used": raw_used,
                    "memory_total": raw_total, "vram_total": raw_total, "mem_total": raw_total,
                    "vram_allocation": raw_used, # Guessing key based on UI label
                    
                    # GB Versions
                    "memory_used_gb": gb_used, "vram_used_gb": gb_used,
                    "memory_total_gb": gb_total, "vram_total_gb": gb_total,
                    
                    "temperature": temp, "temp": temp, "thermal": temp, "gpu_temp": temp
                })
                success_nvml = True
            except Exception as e:
                # NVML transient failure, fall through to PyTorch
                pass

        # ----------------------------------------------------------------------
        # STRATEGY 2: PyTorch Fallback (Secondary)
        # ----------------------------------------------------------------------
        if not success_nvml and torch.cuda.is_available():
            try:
                # Name
                name = torch.cuda.get_device_name(0)
                
                # Memory (Torch returns Bytes)
                raw_used = torch.cuda.memory_allocated(0)
                raw_total = torch.cuda.get_device_properties(0).total_memory
                gb_used = round(raw_used / (1024 ** 3), 1)
                gb_total = round(raw_total / (1024 ** 3), 1)
                
                data.update({
                    "name": name, "gpu_name": name, "device": name,
                    
                    # Raw Bytes
                    "memory_used": raw_used, "vram_used": raw_used, "mem_used": raw_used,
                    "memory_total": raw_total, "vram_total": raw_total, "mem_total": raw_total,
                    "vram_allocation": raw_used,
                    
                    # GB Versions
                    "memory_used_gb": gb_used, "vram_used_gb": gb_used,
                    "memory_total_gb": gb_total, "vram_total_gb": gb_total,
                    
                    # Static/Unknown values for things Torch can't read
                    "load": 0, "gpu_load": 0,
                    "temperature": "--", "temp": "--"
                })
            except Exception:
                data["gpu_name"] = "Unknown CUDA Device"

        return data

    def shutdown(self):
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

# ==============================================================================
# CALLBACKS (REQUIRED FOR TRAINER)
# ==============================================================================

class EnhancedStateCallback(TrainerCallback):
    """
    Updates global app state with training progress.
    """
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
# MODULE INTERFACE (REQUIRED FOR APP.PY)
# ==============================================================================

_monitor_instance = HardwareMonitor()

def get_hardware_status():
    return _monitor_instance.get_status()