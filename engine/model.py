import os
import torch
from transformers import (
    Qwen3VLProcessor, 
    Qwen3VLForConditionalGeneration, 
    BitsAndBytesConfig
)
from peft import (
    get_peft_model, 
    LoraConfig, 
    prepare_model_for_kbit_training
)

# Define local cache directory for models to keep project self-contained
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_CACHE_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Note: Environment variables for HF_HOME are now handled in app.py.

def load_model_and_processor(base_model_id: str, peft_config: LoraConfig):
    """
    Loads the Qwen3-VL model and processor, then applies the LoRA adapter configuration.
    Uses BitsAndBytesConfig for modern 4-bit quantization and prepares model for k-bit training.

    Args:
        base_model_id (str): The HuggingFace model ID.
        peft_config (LoraConfig): The LoRA configuration object.

    Returns:
        tuple: (model, processor) - The LoRA-wrapped model and the tokenizer/processor.
    """
    print(f"[Engine] Loading Processor for: {base_model_id}")
    processor = Qwen3VLProcessor.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        cache_dir=MODEL_CACHE_DIR,
    )
    
    print(f"[Engine] Loading Model (NF4 Quantization) for: {base_model_id}")
    
    # Modern Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_id,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
        cache_dir=MODEL_CACHE_DIR,
    )
    
    # CRITICAL: Prepare model for QLoRA training
    # This enables input gradients (fixing the UserWarning) and casts layers for stability.
    print("[Engine] Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)
    
    print("[Engine] Applying LoRA adapters...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, processor