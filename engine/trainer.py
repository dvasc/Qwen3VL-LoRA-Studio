import os
import shutil
import time
import json
import traceback
from trl import SFTTrainer
from engine import config, data, model, monitoring

class LoraTrainer:
    """
    Orchestrator class for the LoRA training pipeline.
    Connects data, model, configuration, and monitoring components.
    Handles both Training and Validation workflows.
    """
    def __init__(
        self,
        base_model: str,
        dataset_folder_train: str,
        dataset_folder_val: str,
        output_folder: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        state_ref: dict,
        use_thinking: bool = False,
        temp_folder: str = "temp",
    ):
        self.base_model_id = base_model
        self.dataset_folder_train = dataset_folder_train
        self.dataset_folder_val = dataset_folder_val
        self.output_folder = output_folder
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(learning_rate)
        self.state = state_ref
        self.use_thinking = use_thinking
        self.temp_folder = temp_folder

    def log(self, msg: str):
        """Helper to append logs to the global state safely."""
        print(f"[Trainer] {msg}")
        self.state["logs"].append(msg)
        # Keep buffer small to prevent memory leaks in long runs
        if len(self.state["logs"]) > 100:
            self.state["logs"].pop(0)

    def run(self):
        """
        Executes the full training lifecycle in a background thread.
        Updates self.state with progress and status.
        """
        try:
            self.log("Initializing training environment...")
            self.log(f"Dataset Strategy: {'Including Thinking Fields' if self.use_thinking else 'Standard Messages Only'}")
            
            # 1. Load Model & Processor
            peft_config = config.get_lora_config()
            model_obj, processor = model.load_model_and_processor(self.base_model_id, peft_config)
            self.log(f"Model loaded: {self.base_model_id}")

            # 2. Prepare Datasets
            self.log("Loading training dataset...")
            train_dataset = data.load_and_prepare_dataset(
                self.dataset_folder_train, 
                allow_empty=False, 
                use_thinking=self.use_thinking
            )
            num_train = len(train_dataset)
            self.log(f"Training set ready: {num_train} samples.")

            self.log("Loading validation dataset...")
            # We allow empty validation set, but warn the user
            eval_dataset = data.load_and_prepare_dataset(
                self.dataset_folder_val, 
                allow_empty=True, 
                use_thinking=self.use_thinking
            )
            if eval_dataset:
                self.log(f"Validation set ready: {len(eval_dataset)} samples.")
            else:
                self.log("⚠️ No validation data found. Skipping evaluation phase.")

            # 3. Configure Training
            run_name = f"lora_{int(time.time())}"
            training_args = config.get_training_args(
                output_folder=self.output_folder,
                run_name=run_name,
                batch_size=self.batch_size,
                epochs=self.epochs,
                learning_rate=self.lr
            )

            # Calculate total steps for progress tracking
            steps_per_epoch = max(num_train // max(self.batch_size, 1), 1)
            total_steps = steps_per_epoch * max(self.epochs, 1)
            
            self.log(f"Starting run '{run_name}'. Total steps: {total_steps}")

            # 4. Initialize Trainer
            trainer = SFTTrainer(
                model=model_obj,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data.MultimodalCollator(processor),
                callbacks=[monitoring.EnhancedStateCallback(self.state, total_steps)],
            )

            # 5. Execute Training
            trainer.train()
            
            # Check if stopped manually
            if self.state.get("stop_signal", False):
                self.log("⚠️ Training was manually interrupted by user.")
                self.state["status"] = "INTERRUPTED"
            else:
                self.log("Training loop finished successfully.")
                self.state["status"] = "FINISHED"
                self.state["progress"] = 100.0

            # 6. Execute Validation (if dataset exists and not interrupted)
            if eval_dataset and not self.state.get("stop_signal", False):
                self.log("Running final validation...")
                try:
                    metrics = trainer.evaluate()
                    self.log(f"Validation Loss: {metrics.get('eval_loss', 'N/A')}")
                    
                    # Update Global State for UI
                    self.state["val_metrics"] = metrics
                    
                    # Save metrics to disk
                    metrics_path = os.path.join(training_args.output_dir, "validation_results.json")
                    with open(metrics_path, "w") as f:
                        json.dump(metrics, f, indent=2)
                    
                    # Create human-readable log
                    txt_path = os.path.join(training_args.output_dir, "validation_log.txt")
                    with open(txt_path, "w") as f:
                        f.write("VALIDATION RESULTS\n==================\n")
                        for k, v in metrics.items():
                            f.write(f"{k}: {v}\n")
                            
                except Exception as e:
                    self.log(f"Error during validation: {e}")

            # 7. Save Artifacts (Even if interrupted)
            self.log("Saving adapter and processor...")
            final_path = os.path.join(training_args.output_dir, "final_adapter")
            os.makedirs(final_path, exist_ok=True)
            
            trainer.model.save_pretrained(final_path)
            processor.save_pretrained(final_path)

            # 8. Package Output
            self.log("Creating ZIP package...")
            zip_base = os.path.join(self.output_folder, run_name)
            shutil.make_archive(zip_base, "zip", final_path)
            zip_path = zip_base + ".zip"

            # 9. Update Final State
            self.state["output_zip"] = zip_path
            self.log(f"Adapter packaged at: {zip_path}")

        except Exception as e:
            self.state["status"] = "ERROR"
            self.state["error_msg"] = str(e)
            self.log(f"CRITICAL ERROR: {e}")
            self.log(traceback.format_exc())