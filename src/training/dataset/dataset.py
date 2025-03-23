import json
import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from PIL import Image


class SceneryQADataset(Dataset):
    """场景问答数据集类"""
    
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image_path"]
        
        try:
            # 检查图像文件是否存在
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                return self.__getitem__((idx + 1) % len(self.data))
                
            # 打开图像
            img = Image.open(image_path).convert('RGB')
            
            # 构建对话
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": f"Please answer the question about this scene: {item['question']}"}
                    ]
                }
            ]
            
            # 应用对话模板
            inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True
            )
            
            # 处理答案标签
            answer_tokens = self.processor.tokenizer(
                item["answer"],
                return_tensors="pt",
                padding="max_length",
                max_length=512,
                truncation=True
            ).input_ids
            
            # 根据inputs类型处理标签
            if isinstance(inputs, dict) or hasattr(inputs, "items"):
                # 字典类型和BatchFeature类型处理
                if hasattr(inputs, "input_ids"):
                    labels = torch.full_like(inputs.input_ids, -100)
                    labels[:, -answer_tokens.shape[1]:] = answer_tokens
                    inputs["labels"] = labels
                else:
                    # 如果没有input_ids字段，尝试获取第一个张量字段
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor) and v.dim() > 0:
                            labels = torch.full_like(v, -100)
                            labels[:, -answer_tokens.shape[1]:] = answer_tokens
                            inputs["labels"] = labels
                            break
                
                return {k: v.squeeze(0) for k, v in inputs.items()}
            elif isinstance(inputs, torch.Tensor):
                # Tensor类型处理
                labels = torch.full_like(inputs, -100)
                labels[:, -answer_tokens.shape[1]:] = answer_tokens
                
                return {
                    "input_ids": inputs.squeeze(0),
                    "labels": labels.squeeze(0)
                }
            else:
                print(f"错误: 无法处理的inputs类型: {type(inputs)}")
                return self.__getitem__((idx + 1) % len(self.data))
                
        except Exception as e:
            print(f"处理样本时出错 (idx={idx}, image={image_path}): {str(e)}")
            return self.__getitem__((idx + 1) % len(self.data))


def prepare_data(data_path, images_dir, test_mode=False, test_sample_ratio=0.01, max_train_samples=None, max_val_samples=None):
    """
    准备训练和验证数据集
    
    Args:
        data_path: 原始数据集路径
        images_dir: 图像目录路径
        test_mode: 是否为测试模式
        test_sample_ratio: 测试模式下的抽样比例
        max_train_samples: 最大训练样本数
        max_val_samples: 最大验证样本数
        
    Returns:
        train_data: 训练数据
        val_data: 验证数据
        output_train_data: 训练数据输出路径
        output_val_data: 验证数据输出路径
    """
    # 设置输出路径
    output_train_data = "dataset/sft_train_data.json"
    output_val_data = "dataset/sft_val_data.json"
    
    # 确保目录存在
    os.makedirs("dataset", exist_ok=True)
    
    # 设置小规模测试的参数
    TEST_SAMPLE_RATIO = test_sample_ratio if not test_mode else 0.001  # 测试模式下降低抽样比例
    MAX_TRAIN_SAMPLES = max_train_samples if not test_mode else 20  # 测试模式下限制训练样本数
    MAX_VAL_SAMPLES = max_val_samples if not test_mode else 5  # 测试模式下限制验证样本数

    # 加载数据
    df = pd.read_parquet(data_path)
    print(f"原始数据集大小: {len(df)} 条记录")

    # 场景分析
    total_segments = df['segment_id'].unique().tolist()
    print(f"总共 {len(total_segments)} 个唯一场景")

    # 从所有场景中随机抽取构成训练验证集
    sft_segment_count = max(1, int(len(total_segments) * TEST_SAMPLE_RATIO))
    sampled_segment_ids = random.sample(total_segments, sft_segment_count)
    print(f"为SFT抽样了 {len(sampled_segment_ids)} 个场景 (占比{TEST_SAMPLE_RATIO})")

    # 将抽样的场景分为训练集和验证集，比例为7:1
    train_segment_ids, val_segment_ids = train_test_split(sampled_segment_ids, test_size=1/8, random_state=42)
    print(f"训练集场景数: {len(train_segment_ids)}, 验证集场景数: {len(val_segment_ids)}")

    # 过滤数据
    train_df = df[df['segment_id'].isin(train_segment_ids)]
    val_df = df[df['segment_id'].isin(val_segment_ids)]

    print(f"训练集大小: {len(train_df)} 条记录")
    print(f"验证集大小: {len(val_df)} 条记录")
    
    # 准备数据集
    train_data = prepare_dataset(train_df, images_dir, output_train_data, MAX_TRAIN_SAMPLES)
    val_data = prepare_dataset(val_df, images_dir, output_val_data, MAX_VAL_SAMPLES)

    print(f"最终训练集大小: {len(train_data)} 条记录，保存至 {output_train_data}")
    print(f"最终验证集大小: {len(val_data)} 条记录，保存至 {output_val_data}")
    
    return train_data, val_data, output_train_data, output_val_data


def prepare_dataset(df, images_dir, output_path, max_samples=None):
    """
    准备数据集并保存到文件
    
    Args:
        df: 数据帧
        images_dir: 图像目录路径
        output_path: 输出文件路径
        max_samples: 最大样本数
        
    Returns:
        data: 处理后的数据列表
    """
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