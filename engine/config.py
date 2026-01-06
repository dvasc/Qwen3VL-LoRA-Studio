import os
from peft import LoraConfig, TaskType
from trl import SFTConfig

def get_lora_config() -> LoraConfig:
    """
    Generates the specific LoRA configuration for Qwen3-VL fine-tuning.
    
    Returns:
        LoraConfig: The configuration object defining rank, alpha, and target modules.
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", 
            "v_proj", 
            "k_proj", 
            "o_proj", 
            "gate_proj", 
            "up_proj", 
            "down_proj"
        ],
        bias="none",
        inference_mode=False,
    )

def get_training_args(
    output_folder: str,
    run_name: str,
    batch_size: int,
    epochs: int,
    learning_rate: float
) -> SFTConfig:
    """
    Generates the SFTConfig (TrainingArguments) for the training session.

    Args:
        output_folder (str): Base folder for outputs.
        run_name (str): Unique name for this run (used for directory creation).
        batch_size (int): Batch size per device.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate.

    Returns:
        SFTConfig: The HuggingFace TRL training configuration.
    """
    output_dir = os.path.join(output_folder, run_name)
    os.makedirs(output_dir, exist_ok=True)

    return SFTConfig(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,  # Optimized for 12GB VRAM stability
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        fp16=False,  # Qwen3-VL preference
        bf16=False,
        # Log every step to enable granular ETR calculation and frontend updates
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        optim="adamw_8bit",
        max_grad_norm=1.0,
        dataset_text_field=None,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=None,
        packing=False,
        remove_unused_columns=False,
    )