import logging
import atexit
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
        # Scenario 1: Monitor is in error state (Driver missing/broken)
        if not self.available:
            return {
                "available": False,
                "error": self.error_msg or "Unknown Initialization Error"
            }

        # Scenario 2: Monitor is active, fetch data
        try:
            # 1. Name
            name = pynvml.nvmlDeviceGetName(self.handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            # 2. Utilization (Core Load)
            # Returns object with .gpu and .memory attributes
            util_rates = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            gpu_load = util_rates.gpu

            # 3. Memory (VRAM)
            # Returns object with .total, .free, .used attributes (in bytes)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            
            # Convert to GB for frontend display
            vram_used_gb = round(mem_info.used / (1024**3), 1)
            vram_total_gb = round(mem_info.total / (1024**3), 1)

            # 4. Temperature
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
            # If reading fails (e.g., driver crashed), mark as unavailable
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

# ==============================================================================
# SINGLETON INSTANTIATION & LIFECYCLE
# ==============================================================================

# Initialize singleton
monitor = HardwareMonitor()

# Register cleanup on app exit
atexit.register(monitor.shutdown)

def get_hardware_status():
    """Public interface for the API route."""
    return monitor.get_telemetry()

# ==============================================================================
# TRAINING CALLBACKS
# ==============================================================================

class EnhancedStateCallback(TrainerCallback):
    """
    Callback to update the global app state dictionary during training.
    Used by the LoraTrainer in engine/trainer.py.
    """
    def __init__(self, app_state, total_steps):
        self.app_state = app_state
        self.total_steps = total_steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Capture loss
            if "loss" in logs:
                self.app_state["current_loss"] = round(logs["loss"], 4)
            
            # Calculate percentage progress
            if self.total_steps > 0:
                progress = (state.global_step / self.total_steps) * 100
                self.app_state["progress"] = round(progress, 1)