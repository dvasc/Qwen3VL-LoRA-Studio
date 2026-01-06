import time
import math
from transformers import TrainerCallback

# Safe import for NVIDIA management library
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

def get_hardware_status():
    """
    Polls NVIDIA GPU status using pynvml.
    Returns a dict with safe defaults if GPU/driver is unavailable.
    """
    status = {
        "gpu_name": "N/A",
        "utilization": 0,
        "vram_used": 0.0,
        "vram_total": 0.0,
        "temp": 0,
        "available": False
    }
    
    if not PYNVML_AVAILABLE:
        return status

    try:
        pynvml.nvmlInit()
        # Assume single GPU or take the first one for simplicity
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        name = pynvml.nvmlDeviceGetName(handle)
        # Decode bytes if necessary (pynvml behavior varies by version)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
            
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        
        status["gpu_name"] = name
        status["utilization"] = util.gpu
        status["vram_used"] = round(mem.used / 1024**3, 1) # GB
        status["vram_total"] = round(mem.total / 1024**3, 1) # GB
        status["temp"] = temp
        status["available"] = True
        
    except Exception:
        # Fail silently to prevent crashing the main app loop
        pass
        
    return status

class EnhancedStateCallback(TrainerCallback):
    """
    Hooks into HF Trainer to update global state with progress, 
    detailed logs, ETR (Estimated Time Remaining) calculations,
    and handling of the Stop Signal.
    """
    def __init__(self, state_ref, total_steps: int):
        self.state = state_ref
        self.total_steps = max(int(total_steps), 1)
        self.start_time = time.time()
        self.step_start_time = None
        self.avg_step_time = 0.0
        self.ema_alpha = 0.1 # Smoothing factor for ETR moving average
        
        # Initialize ETR in state
        self.state["etr"] = "--:--"
        
    def format_time(self, seconds):
        if seconds is None or seconds < 0: return "--:--"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def on_step_begin(self, args, state, control, **kwargs):
        """Record the start time of the current step."""
        self.step_start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        """Calculate step duration, update ETR, and check for stop signal."""
        
        # 1. Check for User Interrupt
        if self.state.get("stop_signal", False):
            control.should_training_stop = True
            self.state["logs"].append("🛑 Stop signal received. Halting training loop...")
        
        # 2. Calculate timing
        if self.step_start_time:
            step_duration = time.time() - self.step_start_time
            
            # Update Exponential Moving Average (EMA) for stable ETR
            if self.avg_step_time == 0:
                self.avg_step_time = step_duration
            else:
                self.avg_step_time = (self.ema_alpha * step_duration) + ((1 - self.ema_alpha) * self.avg_step_time)
            
            # Calculate ETR
            remaining_steps = self.total_steps - state.global_step
            if remaining_steps > 0:
                etr_seconds = remaining_steps * self.avg_step_time
                self.state["etr"] = self.format_time(etr_seconds)
            else:
                self.state["etr"] = "00:00"

        # 3. Update Progress
        progress = (state.global_step / self.total_steps) * 100.0
        self.state["progress"] = round(progress, 1)
        
        # 4. Update elapsed duration
        elapsed = time.time() - self.start_time
        self.state["duration"] = self.format_time(elapsed)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Format and append new log lines."""
        if logs:
            loss = logs.get("loss", 0.0)
            current_step = state.global_step
            
            step_time_str = f"{self.avg_step_time:.1f}s/it" if self.avg_step_time > 0 else "init..."
            etr_str = self.state.get("etr", "--:--")
            
            msg = (
                f"Step {current_step}/{self.total_steps} | "
                f"Loss: {loss:.4f} | "
                f"{step_time_str} | "
                f"ETR: {etr_str}"
            )
            
            self.state["logs"].append(msg)
            # Keep log buffer manageable
            if len(self.state["logs"]) > 100:
                self.state["logs"].pop(0)