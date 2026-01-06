import os
import base64
import torch
from io import BytesIO
from PIL import Image
from datasets import load_dataset, Dataset as HFDataset

def base64_to_pil(base64_str: str) -> Image.Image:
    """Convert base64 data URI or raw string to PIL Image."""
    try:
        if base64_str.startswith("data:"):
            base64_str = base64_str.split(",", 1)[1]
        image_data = base64.b64decode(base64_str)
        return Image.open(BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {e}")

def extract_text_and_images(messages):
    """
    Extract text and images from message format for validation.
    """
    text_parts = []
    images = []
    
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        text_parts.append(text)
                elif item.get("type") == "image":
                    image_data = item.get("image", "")
                    if isinstance(image_data, str):
                        try:
                            pil_image = base64_to_pil(image_data)
                            images.append(pil_image)
                        except Exception:
                            pass
    
    text = " ".join(text_parts) if text_parts else ""
    return text, images

def validate_sample(sample):
    """Validate that a sample has the required structure."""
    messages = sample.get("messages")
    if not messages or not isinstance(messages, list):
        return False
    
    # Must have at least one message with content list
    for msg in messages:
        if isinstance(msg, dict) and isinstance(msg.get("content"), list):
            return True
    return False

def collect_data_files(dataset_folder: str):
    """Scan folder for JSONL or JSON files."""
    files = []
    if not os.path.exists(dataset_folder):
        return files
        
    for f in os.listdir(dataset_folder):
        if f.endswith(".jsonl") or f.endswith(".json"):
            files.append(os.path.join(dataset_folder, f))
    return files

def transform_reasoning_sample(sample, use_thinking: bool):
    """
    Transforms a sample by merging 'thinking' and 'response' fields into the messages list.
    Includes robustness checks for null/invalid messages.
    """
    # CRITICAL FIX: explicit check for None or non-list messages
    messages = sample.get("messages")
    if messages is None or not isinstance(messages, list):
        # Return as-is; the subsequent filter(validate_sample) will drop it
        return sample

    # Check if we have separate fields to merge
    has_response = "response" in sample
    has_thinking = "thinking" in sample
    
    if not has_response and not has_thinking:
        return sample
    
    # Safe clone of valid messages list
    new_messages = list(messages)
    
    # Determine final content
    final_text = ""
    
    if use_thinking and has_thinking and sample["thinking"]:
        # Wrap thinking in tags
        final_text += f"<think>\n{sample['thinking']}\n</think>\n"
        
    if has_response and sample["response"]:
        final_text += sample["response"]
        
    if not final_text:
        return sample
        
    # Append new assistant message
    new_messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": final_text}]
    })
    
    return {"messages": new_messages}

def load_and_prepare_dataset(dataset_folder: str, allow_empty: bool = False, use_thinking: bool = False):
    """
    Loads, transforms, validates, and prepares the dataset from the folder.
    """
    data_files = collect_data_files(dataset_folder)
    if not data_files:
        if allow_empty:
            return None
        raise RuntimeError(f"No dataset files found in {dataset_folder}")

    # Load via streaming first
    full_dataset = load_dataset(
        "json",
        data_files=data_files,
        split="train",
        streaming=False 
    )
    
    # 1. Transform: Merge thinking/response into messages if they exist
    dataset = full_dataset.map(lambda x: transform_reasoning_sample(x, use_thinking))
    
    # 2. Validate: Ensure structure is correct (drops samples where messages=None)
    dataset = dataset.filter(validate_sample)
    
    # Check length
    if len(dataset) == 0:
        if allow_empty:
            return None
        raise RuntimeError("No valid training samples found after validation.")
    
    # 3. Format: Keep only the messages column
    return HFDataset.from_dict({
        "messages": dataset["messages"]
    })

class MultimodalCollator:
    """
    Custom collator for Qwen3-VL multimodal training.
    """
    def __init__(self, processor):
        self.processor = processor
        self.error_count = 0
        self.success_count = 0
    
    def _process_single_sample(self, messages):
        text, images = extract_text_and_images(messages)
        
        if not text:
            raise ValueError("No text content in sample")
        if not images:
            raise ValueError("No images in sample")
        
        # Build chat messages for processor
        chat_messages = []
        for msg in messages:
            chat_msg = {"role": msg.get("role", "user"), "content": []}
            content = msg.get("content", [])
            
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "image":
                        image_data = item.get("image", "")
                        if isinstance(image_data, str):
                            try:
                                pil_image = base64_to_pil(image_data)
                                chat_msg["content"].append({"type": "image", "image": pil_image})
                            except Exception:
                                pass
                    elif item.get("type") == "text":
                        chat_msg["content"].append({"type": "text", "text": item.get("text", "")})
            
            if chat_msg["content"]:
                chat_messages.append(chat_msg)
        
        if not chat_messages:
            raise ValueError("No valid chat messages after processing")
        
        text_with_template = self.processor.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        processed_images = []
        for msg in chat_messages:
            for item in msg.get("content", []):
                if isinstance(item, dict) and item.get("type") == "image":
                    processed_images.append(item["image"])
        
        processed = self.processor(
            text=text_with_template,
            images=processed_images if processed_images else None,
            return_tensors="pt",
        )
        
        return processed
    
    def __call__(self, batch):
        processed_batch = {
            "input_ids": [],
            "pixel_values": [],
            "attention_mask": [],
            "image_grid_thw": [],
        }
        
        valid_samples = 0
        
        for idx, sample in enumerate(batch):
            messages = sample.get("messages", [])
            if not messages:
                continue
            
            try:
                processed = self._process_single_sample(messages)
                
                input_ids = processed.get("input_ids")
                attention_mask = processed.get("attention_mask")
                pixel_values = processed.get("pixel_values")
                image_grid_thw = processed.get("image_grid_thw")
                
                if input_ids is None or input_ids.numel() == 0:
                    continue
                
                input_ids = input_ids.long().cpu()
                
                if attention_mask is not None:
                    attention_mask = attention_mask.long().cpu()
                else:
                    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
                
                if input_ids.shape != attention_mask.shape:
                    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
                
                processed_batch["input_ids"].append(input_ids)
                processed_batch["attention_mask"].append(attention_mask)
                
                if pixel_values is not None:
                    pixel_values = pixel_values.cpu() if hasattr(pixel_values, 'cpu') else pixel_values
                    processed_batch["pixel_values"].append(pixel_values)
                
                if image_grid_thw is not None:
                    processed_batch["image_grid_thw"].append(image_grid_thw)
                
                valid_samples += 1
                self.success_count += 1
            
            except Exception as e:
                self.error_count += 1
                continue
        
        if valid_samples == 0:
            return {
                "input_ids": torch.tensor([[0]], dtype=torch.long),
                "attention_mask": torch.tensor([[1]], dtype=torch.long),
                "labels": torch.tensor([[0]], dtype=torch.long),
            }
        
        final_batch = {}
        
        if processed_batch["input_ids"]:
            max_len = max(ids.shape[-1] for ids in processed_batch["input_ids"])
            padded_ids = []
            padded_masks = []
            
            for ids, mask in zip(processed_batch["input_ids"], processed_batch["attention_mask"]):
                if ids.dim() == 1: ids = ids.unsqueeze(0)
                if mask.dim() == 1: mask = mask.unsqueeze(0)
                
                pad_len = max_len - ids.shape[-1]
                if pad_len > 0:
                    ids = torch.cat([ids, torch.full((ids.shape[0], pad_len), 0, dtype=torch.long)], dim=-1)
                    mask = torch.cat([mask, torch.zeros((mask.shape[0], pad_len), dtype=torch.long)], dim=-1)
                
                padded_ids.append(ids)
                padded_masks.append(mask)
            
            final_batch["input_ids"] = torch.cat(padded_ids, dim=0).long()
            final_batch["attention_mask"] = torch.cat(padded_masks, dim=0).long()
            final_batch["labels"] = final_batch["input_ids"].clone()
            final_batch["labels"][final_batch["attention_mask"] == 0] = -100

        if processed_batch["pixel_values"]:
            try:
                final_batch["pixel_values"] = torch.cat(processed_batch["pixel_values"], dim=0)
            except Exception:
                if processed_batch["pixel_values"]:
                    final_batch["pixel_values"] = processed_batch["pixel_values"][0]
        
        if processed_batch["image_grid_thw"]:
            try:
                stacked_grids = []
                for grid in processed_batch["image_grid_thw"]:
                    if isinstance(grid, torch.Tensor):
                        if grid.dim() == 1: grid = grid.unsqueeze(0)
                        stacked_grids.append(grid)
                if stacked_grids:
                    final_batch["image_grid_thw"] = torch.cat(stacked_grids, dim=0)
            except Exception:
                pass
        
        return final_batch