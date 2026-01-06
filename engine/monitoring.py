import pynvml
import time
import logging

# Configure logger for this module
logger = logging.getLogger(__name__)

class HardwareMonitor:
    def __init__(self):
        self.nvml_initialized = False
        try:
            pynvml.nvmlInit()
            self.nvml_initialized = True
            logger.info("NVML Initialized Successfully.")
        except pynvml.NVMLError as e:
            logger.error(f"Failed to initialize NVML: {e}. Hardware monitoring will be disabled.")
        except Exception as e:
            logger.error(f"Unexpected error initializing NVML: {e}")

    def get_status(self):
        """
        Retrieves real-time hardware telemetry from the GPU.
        Returns a dictionary with safe default values if drivers/hardware are unavailable.
        """
        status = {
            "gpu_name": "N/A",
            "gpu_load": 0,
            "memory_used": 0,
            "memory_total": 0,
            "temperature": "--"
        }

        if not self.nvml_initialized:
            # Attempt re-initialization once if it failed previously
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
            except Exception:
                return status

        try:
            # Monitor primary GPU (index 0)
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # 1. GPU Name
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            status["gpu_name"] = name

            # 2. Utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            status["gpu_load"] = utilization.gpu

            # 3. Memory Info
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # Convert bytes to GB
            status["memory_used"] = round(memory_info.used / (1024 ** 3), 1)
            status["memory_total"] = round(memory_info.total / (1024 ** 3), 1)

            # 4. Temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            status["temperature"] = temp

        except pynvml.NVMLError as e:
            logger.warning(f"NVML Error during status poll: {e}")
            status["gpu_name"] = "Driver Error"
        except Exception as e:
            logger.error(f"Unexpected error during status poll: {e}")
            status["gpu_name"] = "System Error"

        return status

    def shutdown(self):
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

# ==============================================================================
# MODULE INTERFACE (Fixes ImportError)
# ==============================================================================

# Global singleton instance
_monitor_instance = HardwareMonitor()

def get_hardware_status():
    """
    Public API function expected by app.py.
    Delegates to the singleton HardwareMonitor instance.
    """
    return _monitor_instance.get_status()