import logging
import atexit
import time
from transformers import TrainerCallback

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HardwareMonitor")

# Try to import pynvml; handle the case where it's missing entirely
try:
    import pynvml
    PYNVML_INSTALLED = True
except ImportError:
    PYNVML_INSTALLED = False
    logger.error("❌ pynvml not installed. Hardware monitoring will be disabled.")

class HardwareMonitor:
    """
    Singleton class to manage NVIDIA GPU telemetry via NVML.
    Provides cross-platform monitoring for GCE (Linux) and Local (Windows).
    """
    def __init__(self):
        self.available = False
        self.handle = None
        self.error_msg = None
        self.device_index = 0  # Default to primary GPU
        
        # Attempt initialization immediately upon instantiation
        self._initialize_nvml()

    def _initialize_nvml(self):
        """
        Attempts to load the NVML driver and acquire a handle to the GPU.
        """
        if not PYNVML_INSTALLED:
            self.error_msg = "pynvml library not found in environment."
            return

        try:
            pynvml.nvmlInit()
            # Get handle for the first GPU
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self.available = True
            
            # Log success with device name
            name = pynvml.nvmlDeviceGetName(self.handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            logger.info(f"✅ NVML Initialized. Monitoring GPU: {name}")
            
        except pynvml.NVMLError as e:
            self.available = False
            self.error_msg = self._format_nvml_error(e)
            logger.warning(f"⚠️ NVML Initialization Failed: {self.error_msg}")
        except Exception as e:
            self.available = False
            self.error_msg = f"Unexpected Error: {str(e)}"
            logger.error(f"⚠️ Critical Hardware Monitor Error: {self.error_msg}")

    def _format_nvml_error(self, err):
        """Helper to make NVML errors human-readable."""
        msg = str(err)
        if "LibraryNotFound" in msg:
            return "NVIDIA Driver Not Found"
        if "DriverNotLoaded" in msg:
            return "NVIDIA Driver Not Loaded"
        if "NoDevice" in msg:
            return "No NVIDIA GPU Detected"
        return msg

    def get_telemetry(self):
        """
        Retrieves real-time metrics from the GPU.
        Returns a dictionary with 'available' boolean and metrics or error info.
        """
        if not self.available:
            return {
                "available": False,
                "error": self.error_msg or "Unknown Initialization Error"
            }

        try:
            name = pynvml.nvmlDeviceGetName(self.handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            util_rates = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            gpu_load = util_rates.gpu

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            vram_used_gb = round(mem_info.used / (1024**3), 1)
            vram_total_gb = round(mem_info.total / (1024**3), 1)

            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)

            return {
                "available": True,
                "gpu_name": name,
                "utilization": gpu_load,
                "vram_used": vram_used_gb,
                "vram_total": vram_total_gb,
                "temp": temp
            }
        except pynvml.NVMLError as e:
            self.available = False
            self.error_msg = f"Runtime Error: {self._format_nvml_error(e)}"
            return {
                "available": False,
                "error": self.error_msg
            }

    def shutdown(self):
        """Cleanly releases NVML resources."""
        if self.available and PYNVML_INSTALLED:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self.available = False

monitor = HardwareMonitor()
atexit.register(monitor.shutdown)

def get_hardware_status():
    """Public interface for the API route."""
    return monitor.get_telemetry()

def _format_time(seconds):
    """Formats seconds into HH:MM:SS or MM:SS."""
    if seconds is None:
        return "--:--"
    
    s = int(seconds)
    h, remainder = divmod(s, 3600)
    m, s = divmod(remainder, 60)
    
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}"

class EnhancedStateCallback(TrainerCallback):
    """
    Callback to update the global app state dictionary during training.
    """
    def __init__(self, app_state, total_steps):
        self.app_state = app_state
        self.total_steps = total_steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or self.app_state["status"] != "TRAINING":
            return

        # --- Metrics Logging to UI ---
        log_line = ""
        # Case 1: Standard training step log
        if "loss" in logs and "learning_rate" in logs:
            step = state.global_step
            total = self.total_steps
            
            loss_val = f"loss={logs['loss']:.4f}"
            lr_val = f"lr={logs['learning_rate']:.2e}"
            grad_norm_val = f"grad_norm={logs.get('grad_norm', 0.0):.2f}"
            
            token_acc_val = ""
            if 'mean_token_accuracy' in logs:
                acc = logs['mean_token_accuracy'] * 100
                token_acc_val = f", token_acc={acc:.2f}%"

            log_line = f"Step {step}/{total}: {loss_val}, {grad_norm_val}, {lr_val}{token_acc_val}"

        # Case 2: Final training summary log
        elif "train_loss" in logs and "train_runtime" in logs:
            final_loss = f"final_loss={logs['train_loss']:.4f}"
            runtime = f"runtime={_format_time(logs['train_runtime'])}"
            log_line = f"Training Complete: {final_loss}, {runtime}"
        
        if log_line:
            self.app_state["logs"].append(log_line)
            # Keep buffer small to prevent memory leaks
            if len(self.app_state["logs"]) > 100:
                self.app_state["logs"].pop(0)
        
        # --- State Updates (Progress, Timers) ---
        if "loss" in logs:
            self.app_state["current_loss"] = round(logs["loss"], 4)
        if self.total_steps > 0:
            progress = (state.global_step / self.total_steps) * 100
            self.app_state["progress"] = round(progress, 1)

        start_time = self.app_state.get("start_time")
        if start_time:
            # NOTE: Live elapsed duration is now handled client-side in JS
            # We only calculate ETR here.
            elapsed_seconds = time.time() - start_time
            if state.global_step > 0:
                time_per_step = elapsed_seconds / state.global_step
                remaining_steps = self.total_steps - state.global_step
                etr_seconds = remaining_steps * time_per_step
                self.app_state["etr"] = _format_time(etr_seconds)
            else:
                self.app_state["etr"] = "--:--"