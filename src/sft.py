import json
import random
import os
import pandas as pd
from datetime import datetime

os.makedirs("output/checkpoints", exist_ok=True)
os.makedirs("output/logs", exist_ok=True)
os.makedirs("output/final_model", exist_ok=True)

data_path = "/cloud/data/scenery/train.parquet"
images_dir = "/cloud/data/scenery/images/train"
output_sft_data = "data/sft_0.1_percent_data.json"

df = pd.read_parquet(data_path)
print(f"Original dataset size: {len(df)} records")

total_segments = df['segment_id'].unique().tolist()
print(f"Total {len(total_segments)} unique scenes")

sft_segment_count = max(1, int(len(total_segments) * 0.001))
sampled_segment_ids = random.sample(total_segments, sft_segment_count)
print(f"Sampled {len(sampled_segment_ids)} scenes for SFT training (0.1%)")

sft_df = df[df['segment_id'].isin(sampled_segment_ids)]
print(f"SFT dataset size: {len(sft_df)} records")

sft_data = []
for _, row in sft_df.iterrows():
    segment_id = row['segment_id']
    image_path = os.path.join(images_dir, segment_id, "0.jpg")
    if os.path.exists(image_path):
        sft_data.append({
            "question_id": row['question_id'],
            "segment_id": segment_id,
            "image_path": image_path,
            "question": row['question'],
            "answer": row['answer']
        })

print(f"Final SFT dataset size: {len(sft_data)} records")

os.makedirs(os.path.dirname(output_sft_data), exist_ok=True)
with open(output_sft_data, "w") as f:
    json.dump(sft_data, f, indent=2)

print(f"Sampling completed, data saved to {output_sft_data}")

import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

model_name = "Qwen/Qwen2-VL-2B-Instruct"
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("output", "checkpoints", run_name)

processor = AutoProcessor.from_pretrained(model_name)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    ),
    torch_dtype=torch.float16,
)

model.config.use_cache = False
model.enable_input_require_grads()

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["vision_tower", "vision_proj"],
)

model = get_peft_model(model, lora_config)

trainable_params = 0
all_param = 0
for _, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
print(
    f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
)

for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True

from torch.utils.data import Dataset

class SceneryQADataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image_path"]
        
        try:
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                return self.__getitem__((idx + 1) % len(self.data))
                
            from PIL import Image
            try:
                img = Image.open(image_path).convert('RGB')
            except Exception as img_err:
                print(f"Invalid image file {image_path}: {str(img_err)}")
                return self.__getitem__((idx + 1) % len(self.data))
            
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": f"Please answer the question about this scene: {item['question']}"}
                    ]
                }
            ]
            
            inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True
            )
            
            answer_tokens = self.processor.tokenizer(
                item["answer"],
                return_tensors="pt",
                padding="max_length",
                max_length=512,
                truncation=True
            ).input_ids
            
            labels = torch.full_like(inputs.input_ids, -100)
            labels[:, -answer_tokens.shape[1]:] = answer_tokens
            inputs["labels"] = labels
            
            return {k: v.squeeze(0) for k, v in inputs.items()}
        except Exception as e:
            print(f"Error processing sample (idx={idx}, image={image_path}): {str(e)}")
            return self.__getitem__((idx + 1) % len(self.data))

with open(output_sft_data, 'r') as f:
    sft_data = json.load(f)

dataset = SceneryQADataset(sft_data, processor)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir=output_dir,
    run_name=run_name,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    evaluation_strategy="no",
    save_strategy="epoch",
    logging_dir=os.path.join("output", "logs", run_name),
    logging_steps=10,
    fp16=True,
    save_total_limit=3,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

try:
    trainer.train()
    training_success = True
except Exception as e:
    print(f"Training failed: {str(e)}")
    training_success = False

if training_success:
    final_model_dir = os.path.join("output", "final_model", run_name)
    trainer.save_model(final_model_dir)

    # 将配置转换为可JSON序列化的格式
    def convert_to_serializable(obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        else:
            return obj
    
    # 获取配置并转换
    lora_dict = convert_to_serializable(lora_config.to_dict())
    training_dict = convert_to_serializable(training_args.to_dict())
    
    config = {
        "model_name": model_name,
        "lora_config": lora_dict,
        "training_args": training_dict,
        "dataset_info": {
            "total_samples": len(dataset),
            "data_path": output_sft_data,
            "original_data_path": data_path,
            "images_dir": images_dir,
            "sampling_rate": "0.1%",
            "images_per_scene": 1
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
else:
    print("Training did not complete successfully, final model not saved.")
