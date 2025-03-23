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

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 确保目录存在
os.makedirs("output/checkpoints", exist_ok=True)
os.makedirs("output/logs", exist_ok=True)
os.makedirs("output/final_model", exist_ok=True)

# 导入自定义模块
from dataset.dataset import prepare_data, SceneryQADataset
from models.model import setup_model, prepare_for_training, generate_predictions
from models.judge import SceneryJudge
from evaluation.evaluate import compute_metrics, evaluate_with_judge, EvaluationCallback, calculate_model_perplexity
from utils.trainer import MemoryEfficientTrainer, setup_training_args, save_training_results


def main():
    """
    SFT训练主函数
    """
    # 设置路径和参数
    data_path = "/cloud/data/scenery/train.parquet"
    images_dir = "/cloud/data/scenery/images/train"
    
    # 测试模式设置
    TEST_MODE = True  # 测试模式标志
    
    # 准备数据
    train_data, val_data, output_train_data, output_val_data = prepare_data(
        data_path=data_path,
        images_dir=images_dir,
        test_mode=TEST_MODE
    )
    
    # 设置模型
    model, processor = setup_model()
    model, lora_config = prepare_for_training(model)
    
    # 检查可用的GPU数量
    num_gpus = torch.cuda.device_count()
    
    # 创建数据集
    train_dataset = SceneryQADataset(train_data, processor)
    val_dataset = SceneryQADataset(val_data, processor)
    
    print(f"训练数据集大小: {len(train_dataset)} 样本")
    print(f"验证数据集大小: {len(val_dataset)} 样本")
    
    # 设置训练参数
    training_args, run_name = setup_training_args(
        test_mode=TEST_MODE,
        num_gpus=num_gpus
    )
    
    # 清理显存
    torch.cuda.empty_cache()
    
    # 创建训练器
    trainer = MemoryEfficientTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # 初始化评测模型
    try:
        # 选择第二个GPU进行评估，如果可用
        judge_device = torch.device("cuda:1" if num_gpus > 1 else "cuda:0")
        judge_model = SceneryJudge().to(judge_device)
        print(f"成功初始化评测模型，使用设备: {judge_device}")
    except Exception as e:
        print(f"初始化评测模型失败: {str(e)}")
        judge_model = None
    
    # 添加自定义评估回调
    evaluation_callback = EvaluationCallback(
        trainer, 
        val_dataset, 
        processor, 
        judge_model,
        test_mode=TEST_MODE,
        num_gpus=num_gpus
    )
    trainer.add_callback(evaluation_callback)
    
    # 在训练前进行一次评估作为baseline
    print("开始基准评估 (训练前)...")
    
    # 首先计算base model的perplexity loss
    try:
        baseline_loss = calculate_model_perplexity(
            model, 
            val_dataset, 
            processor, 
            batch_size=1, 
            max_samples=5 if TEST_MODE else None
        )
    except Exception as e:
        print(f"计算base model loss失败: {str(e)}")
        baseline_loss = 0.0  # 如果计算失败，使用默认值
    
    # 使用evaluate_with_judge函数进行评估
    baseline_judge_metrics = evaluate_with_judge(
        model, 
        val_dataset, 
        processor, 
        trainer, 
        judge_model,
        num_gpus=num_gpus,
        test_mode=TEST_MODE
    )
    
    # 合并指标，使用实际计算的loss而不是固定值
    baseline_combined_metrics = {**{"loss": baseline_loss}, **baseline_judge_metrics}
    print(f"综合基准评估指标: {baseline_combined_metrics}")
    
    # 开始训练
    try:
        # 确保训练前清理显存
        torch.cuda.empty_cache()
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
        final_judge_metrics = evaluate_with_judge(
            model, 
            val_dataset, 
            processor, 
            trainer, 
            judge_model,
            num_gpus=num_gpus,
            test_mode=TEST_MODE
        )
        final_combined_metrics = {**final_metrics, **final_judge_metrics}
        print(f"综合最终评估指标: {final_combined_metrics}")
        
        # 保存最终结果
        final_model_dir = os.path.join("output", "final_model", run_name)
        save_training_results(
            trainer=trainer,
            model=model,
            final_model_dir=final_model_dir,
            run_name=run_name,
            lora_config=lora_config,
            training_args=training_args,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            baseline_metrics=baseline_combined_metrics,
            final_metrics=final_combined_metrics,
            eval_results=evaluation_callback.eval_results,
            train_data_path=output_train_data,
            val_data_path=output_val_data,
            data_path=data_path,
            images_dir=images_dir,
            test_sample_ratio=0.001 if TEST_MODE else 0.01
        )
    else:
        print("训练未成功完成，未保存最终模型。")


if __name__ == "__main__":
    main()
