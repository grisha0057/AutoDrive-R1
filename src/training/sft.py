import json
import random
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from typing import List
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Trainer, TrainingArguments, BitsAndBytesConfig, AutoModelForSequenceClassification, TrainerCallback
from peft import LoraConfig, get_peft_model

os.makedirs("output/checkpoints", exist_ok=True)
os.makedirs("output/logs", exist_ok=True)
os.makedirs("output/final_model", exist_ok=True)

data_path = "/cloud/data/scenery/train.parquet"
images_dir = "/cloud/data/scenery/images/train"
output_train_data = "data/sft_train_data.json"
output_val_data = "data/sft_val_data.json"

# 设置小规模测试的参数
TEST_MODE = True  # 测试模式标志
TEST_SAMPLE_RATIO = 0.001 if TEST_MODE else 0.01  # 测试模式下降低抽样比例为原来的1/10
MAX_TRAIN_SAMPLES = 20 if TEST_MODE else None  # 测试模式下限制训练样本数
MAX_VAL_SAMPLES = 5 if TEST_MODE else None  # 测试模式下限制验证样本数
TEST_EPOCHS = 0.1 if TEST_MODE else 1  # 测试模式下只训练0.1个epoch

df = pd.read_parquet(data_path)
print(f"原始数据集大小: {len(df)} 条记录")

total_segments = df['segment_id'].unique().tolist()
print(f"总共 {len(total_segments)} 个唯一场景")

# 从所有场景中随机抽取 TEST_SAMPLE_RATIO 构成训练验证集
sft_segment_count = max(1, int(len(total_segments) * TEST_SAMPLE_RATIO))
sampled_segment_ids = random.sample(total_segments, sft_segment_count)
print(f"为SFT抽样了 {len(sampled_segment_ids)} 个场景 (占比{TEST_SAMPLE_RATIO})")

# 将抽样的场景分为训练集和验证集，比例为7:1
train_segment_ids, val_segment_ids = train_test_split(sampled_segment_ids, test_size=1/8, random_state=42)
print(f"训练集场景数: {len(train_segment_ids)}, 验证集场景数: {len(val_segment_ids)}")

sft_df = df[df['segment_id'].isin(sampled_segment_ids)]
train_df = df[df['segment_id'].isin(train_segment_ids)]
val_df = df[df['segment_id'].isin(val_segment_ids)]

print(f"SFT数据集大小: {len(sft_df)} 条记录")
print(f"训练集大小: {len(train_df)} 条记录")
print(f"验证集大小: {len(val_df)} 条记录")

def prepare_dataset(df, output_path, max_samples=None):
    data = []
    for _, row in df.iterrows():
        if max_samples is not None and len(data) >= max_samples:
            break
            
        segment_id = row['segment_id']
        image_path = os.path.join(images_dir, segment_id, "0.jpg")
        if os.path.exists(image_path):
            data.append({
                "question_id": row['question_id'],
                "segment_id": segment_id,
                "image_path": image_path,
                "question": row['question'],
                "answer": row['answer']
            })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    return data

train_data = prepare_dataset(train_df, output_train_data, MAX_TRAIN_SAMPLES)
val_data = prepare_dataset(val_df, output_val_data, MAX_VAL_SAMPLES)

print(f"最终训练集大小: {len(train_data)} 条记录，保存至 {output_train_data}")
print(f"最终验证集大小: {len(val_data)} 条记录，保存至 {output_val_data}")

model_name = "Qwen/Qwen2-VL-2B-Instruct"
run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_test" if TEST_MODE else datetime.now().strftime("%Y%m%d_%H%M%S")
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
    f"可训练参数: {trainable_params} || 总参数: {all_param} || 可训练比例: {100 * trainable_params / all_param}"
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
            print(f"处理样本时出错 (idx={idx}, image={image_path}): {str(e)}")
            return self.__getitem__((idx + 1) % len(self.data))

# 添加评测模型，参考judge.py
class SceneryJudge(nn.Module):
    """
    SceneryJudge是一个文本分类器，用于评估场景问答的准确性。
    """
    def __init__(self, pretrained_model="Qwen/Qwen2-VL-2B-Instruct"):
        super().__init__()
        self.tokenizer = AutoProcessor.from_pretrained(pretrained_model)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            pretrained_model
        ).eval()

    @torch.inference_mode()
    def forward(self, question: str, references: List[str], prediction: str):
        """
        评估模型的推理函数，通过比较生成文本与参考答案的相似度评分。
        Args:
            question: 输入问题。
            references: 参考答案列表。
            prediction: 模型预测。
        Output:
            scores: 指示准确性的分数。
        """
        device = next(self.parameters()).device
        texts = [
            f"问题: {question}\n参考答案: {a_gt}\n预测: {prediction}"
            for a_gt in references
        ]

        # 使用生成模型计算困惑度或相似度作为评分
        scores = []
        for text in texts:
            encoded_input = self.tokenizer.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            with torch.no_grad():
                outputs = self.model(**encoded_input, labels=encoded_input["input_ids"])
                # 使用模型的损失作为评分指标（损失越低，评分越高）
                loss = outputs.loss
                # 转换为分数（损失越低，分数越高）
                score = torch.exp(-loss)
            scores.append(score.item())
        return torch.tensor(scores)

    def compute(self, questions: List[str], references: List[List[str]], predictions: List[str]):
        """
        计算评估指标。对于多个参考答案，选择最高分数。
        Args:
            questions: 输入问题列表。
            references: 参考答案列表的列表，每个问题支持多个参考答案。
            predictions: 模型预测列表。
        Output:
            scores: 指示准确性的分数。
        """
        max_scores = []

        for index, question in enumerate(questions):
            references_preprocessed = [self.preprocess(reference) for reference in references[index]]
            prediction_preprocessed = self.preprocess(predictions[index])
            scores = self.forward(question, references_preprocessed, prediction_preprocessed)
            max_score = max(scores).item()
            max_scores.append(max_score)
        return torch.tensor(max_scores)
        
    def preprocess(self, string: str):
        """
        一致性预处理函数。
        Args:
            string: 要处理的输入字符串。
        Output:
            output: 处理后的字符串，小写并去除前后空格。
        """
        output = str(string).lower().lstrip().rstrip() 
        return output

# 添加预测生成函数
def generate_predictions(model, eval_dataset, processor, device):
    """
    使用模型为验证集生成预测结果
    """
    model.eval()
    predictions = []
    questions = []
    references = []
    
    for i in range(len(eval_dataset)):
        item = eval_dataset.data[i]
        image_path = item["image_path"]
        question = item["question"]
        reference = item["answer"]
        
        try:
            from PIL import Image
            img = Image.open(image_path).convert('RGB')
            
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": f"请回答关于这个场景的问题: {question}"}
                    ]
                }
            ]
            
            inputs = processor.apply_chat_template(
                conversation,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=64 if TEST_MODE else 256,  # 减少生成的token数量
                    do_sample=False
                )
            
            generated_text = processor.tokenizer.decode(output[0], skip_special_tokens=True)
            prediction = generated_text.split("assistant:")[-1].strip()
            
            predictions.append(prediction)
            questions.append(question)
            references.append([reference])
        except Exception as e:
            print(f"生成预测时出错 (idx={i}, image={image_path}): {str(e)}")
            # 在出错情况下添加空值，保持索引一致
            predictions.append("")
            questions.append(question)
            references.append([reference])
    
    return questions, references, predictions

# 自定义验证函数
def compute_metrics(eval_pred):
    """
    计算评估指标
    
    包括语言模型交叉熵损失和自定义评测模型的指标
    """
    logits, labels = eval_pred
    
    # 计算交叉熵损失
    loss = np.mean(np.where(labels != -100, (logits - labels)**2, 0))
    
    metrics = {
        "loss": loss,
        # 注意: 由于评测模型需要生成预测结果，这里无法直接计算
        # 评测模型的指标将在evaluate_with_judge函数中计算
    }
    
    return metrics

# 自定义评估函数，使用评测模型
def evaluate_with_judge(model, eval_dataset, processor, trainer, judge_model=None):
    """
    使用评测模型对验证集进行评估
    """
    print("开始使用评测模型评估...")
    
    # 将模型移到评估模式
    model.eval()
    device = next(model.parameters()).device
    
    # 初始化评测模型（如果未提供）
    if judge_model is None:
        try:
            judge_model = SceneryJudge().to(device)
            print("成功初始化评测模型")
        except Exception as e:
            print(f"初始化评测模型失败: {str(e)}")
            return {"judge_score": 0.0}
    
    # 生成预测结果
    questions, references, predictions = generate_predictions(model, eval_dataset, processor, device)
    
    # 使用评测模型评估
    try:
        scores = judge_model.compute(questions, references, predictions)
        judge_score = scores.mean().item()
        print(f"评测模型评分: {judge_score:.4f}")
        
        # 返回指标结果
        return {
            "judge_score": judge_score,
            "judge_scores": scores.tolist()
        }
    except Exception as e:
        print(f"评测模型评估失败: {str(e)}")
        return {"judge_score": 0.0}

# 自定义评估回调
class EvaluationCallback(TrainerCallback):
    def __init__(self, trainer, eval_dataset, processor, judge_model=None):
        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.judge_model = judge_model
        self.best_judge_score = 0.0
        self.eval_results = []
    
    def on_step_end(self, args, state, control, **kwargs):
        # 测试模式下减少评估频率，每100步评估一次
        eval_frequency = 100 if TEST_MODE else 400
        
        if state.global_step > 0 and state.global_step % eval_frequency == 0:
            print(f"\n在步骤 {state.global_step} 执行评估...")
            
            # 常规评估
            metrics = self.trainer.evaluate()
            print(f"评估指标: {metrics}")
            
            # 使用评测模型的评估
            judge_metrics = evaluate_with_judge(
                self.trainer.model, 
                self.eval_dataset, 
                self.processor,
                self.trainer,
                self.judge_model
            )
            
            # 合并指标
            combined_metrics = {**metrics, **judge_metrics}
            print(f"综合评估指标: {combined_metrics}")
            
            # 保存评估结果
            self.eval_results.append({
                "step": state.global_step,
                "metrics": combined_metrics
            })
            
            # 更新最佳分数
            if judge_metrics.get("judge_score", 0) > self.best_judge_score:
                self.best_judge_score = judge_metrics.get("judge_score", 0)
                print(f"新的最佳评测模型分数: {self.best_judge_score:.4f}")
            
            print(f"评估完成。当前最佳评测模型分数: {self.best_judge_score:.4f}")

# 主训练逻辑开始
train_dataset = SceneryQADataset(train_data, processor)
val_dataset = SceneryQADataset(val_data, processor)

print(f"训练数据集大小: {len(train_dataset)} 样本")
print(f"验证数据集大小: {len(val_dataset)} 样本")

from transformers import TrainingArguments, Trainer, TrainerCallback
import numpy as np
# import evaluate

# 测试模式下自定义步数和周期
eval_steps = 50 if TEST_MODE else 400
save_steps = 100 if TEST_MODE else 400

training_args = TrainingArguments(
    output_dir=output_dir,
    run_name=run_name,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1 if TEST_MODE else 2,  # 测试模式下减少梯度累积步数
    num_train_epochs=TEST_EPOCHS,  # 测试模式下减少训练epoch数
    learning_rate=1e-4,
    eval_strategy="steps",  # 按步数评估
    eval_steps=eval_steps,  # 评估步数
    save_strategy="steps",
    save_steps=save_steps,  # 保存步数
    logging_dir=os.path.join("output", "logs", run_name),
    logging_steps=10,
    fp16=True,
    save_total_limit=2 if TEST_MODE else 3,  # 测试模式下减少保存的检查点数量
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    load_best_model_at_end=True,
    metric_for_best_model="judge_score",  # 使用评测模型分数作为选择最佳模型的指标
    greater_is_better=True,  # 评测分数越高越好
    max_steps=200 if TEST_MODE else -1,  # 测试模式下限制最大步数
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# 初始化评测模型
try:
    device = next(model.parameters()).device
    judge_model = SceneryJudge().to(device)
    print("成功初始化评测模型")
except Exception as e:
    print(f"初始化评测模型失败: {str(e)}")
    judge_model = None

# 添加自定义评估回调
evaluation_callback = EvaluationCallback(trainer, val_dataset, processor, judge_model)
trainer.add_callback(evaluation_callback)

# 在训练前进行一次评估作为baseline
print("开始基准评估 (训练前)...")
baseline_metrics = trainer.evaluate()
print(f"基准评估指标: {baseline_metrics}")

# 使用评测模型进行基准评估
baseline_judge_metrics = evaluate_with_judge(model, val_dataset, processor, trainer, judge_model)
baseline_combined_metrics = {**baseline_metrics, **baseline_judge_metrics}
print(f"综合基准评估指标: {baseline_combined_metrics}")

try:
    trainer.train()
    training_success = True
except Exception as e:
    print(f"训练失败: {str(e)}")
    training_success = False

if training_success:
    # 训练完成后进行最终评估
    print("开始最终评估...")
    final_metrics = trainer.evaluate()
    print(f"最终评估指标: {final_metrics}")
    
    # 使用评测模型进行最终评估
    final_judge_metrics = evaluate_with_judge(model, val_dataset, processor, trainer, judge_model)
    final_combined_metrics = {**final_metrics, **final_judge_metrics}
    print(f"综合最终评估指标: {final_combined_metrics}")
    
    final_model_dir = os.path.join("output", "final_model", run_name)
    trainer.save_model(final_model_dir)

    def convert_to_serializable(obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        else:
            return obj
    
    lora_dict = convert_to_serializable(lora_config.to_dict())
    training_dict = convert_to_serializable(training_args.to_dict())
    
    # 获取所有中间评估结果
    eval_history = convert_to_serializable(evaluation_callback.eval_results)
    
    config = {
        "model_name": model_name,
        "lora_config": lora_dict,
        "training_args": training_dict,
        "dataset_info": {
            "total_samples": len(train_dataset) + len(val_dataset),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "train_data_path": output_train_data,
            "val_data_path": output_val_data,
            "original_data_path": data_path,
            "images_dir": images_dir,
            "sampling_rate": TEST_SAMPLE_RATIO,
            "train_val_split": "7:1"
        },
        "metrics": {
            "baseline": baseline_combined_metrics,
            "final": final_combined_metrics,
            "evaluation_history": eval_history
        }
    }

    with open(os.path.join(final_model_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"""
    训练完成!
    - 最终模型保存至: {final_model_dir}
    - 训练日志: {training_args.logging_dir}
    - 检查点: {output_dir}
    - 训练集大小: {len(train_dataset)} 样本
    - 验证集大小: {len(val_dataset)} 样本
    - 基准评估指标: {baseline_combined_metrics}
    - 最终评估指标: {final_combined_metrics}
    - 最佳评测分数: {evaluation_callback.best_judge_score:.4f}
    """)
else:
    print("训练未成功完成，未保存最终模型。")
