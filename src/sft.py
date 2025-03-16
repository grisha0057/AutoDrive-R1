import json
import random
import os
from datetime import datetime

# Create necessary directories
os.makedirs("data/images", exist_ok=True)
os.makedirs("data/annotations", exist_ok=True)
os.makedirs("output/checkpoints", exist_ok=True)
os.makedirs("output/logs", exist_ok=True)
os.makedirs("output/final_model", exist_ok=True)

# Load dataset
data_path = "data/annotations/lingoqa_dataset.json"
output_sft_data = "data/annotations/sft_5_percent_data.json"

with open(data_path, 'r') as f:
    data = json.load(f)

# Sample 5% of scenes
total_scenes = list(set(entry["scene_id"] for entry in data))
sft_scene_count = int(len(total_scenes) * 0.05)
sampled_scene_ids = set(random.sample(total_scenes, sft_scene_count))

sft_data = [entry for entry in data if entry["scene_id"] in sampled_scene_ids]

with open(output_sft_data, "w") as f:
    json.dump(sft_data, f, indent=2)

print(f"Sampling completed, data saved to {output_sft_data}")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# Model configuration
model_name = "Qwen/Qwen2-VL-2B-Instruct"
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("output", "checkpoints", run_name)

processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ),
)

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

from torch.utils.data import Dataset

class LingoQADataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor
        self.image_dir = "data/images"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, item["image_path"])
        inputs = self.processor(
            text=item["question"],
            images=image_path,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )
        labels = self.processor.tokenizer(item["answer"], return_tensors="pt").input_ids
        inputs["labels"] = labels
        return {k: v.squeeze(0) for k, v in inputs.items()}

dataset = LingoQADataset(json.load(open(output_sft_data)), processor)

from transformers import TrainingArguments, Trainer

# Training configuration
training_args = TrainingArguments(
    output_dir=output_dir,
    run_name=run_name,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=os.path.join("output", "logs", run_name),
    logging_steps=10,
    fp16=True,
    save_total_limit=3,  # Keep only the last 3 checkpoints
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Start training
trainer.train()

# Save final model and training info
final_model_dir = os.path.join("output", "final_model", run_name)
trainer.save_model(final_model_dir)

# Save training configuration
config = {
    "model_name": model_name,
    "lora_config": lora_config.to_dict(),
    "training_args": training_args.to_dict(),
    "dataset_info": {
        "total_samples": len(dataset),
        "data_path": output_sft_data
    }
}

with open(os.path.join(final_model_dir, "training_config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"""
Training completed!
- Final model saved to: {final_model_dir}
- Training logs: {training_args.logging_dir}
- Checkpoints: {output_dir}
""")
