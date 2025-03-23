import os
import torch
import json
from datetime import datetime
from transformers import TrainingArguments, Trainer


class MemoryEfficientTrainer(Trainer):
    """
    内存高效的Trainer，优化显存使用
    """
    def evaluation_loop(self, *args, **kwargs):
        # 在评估前清理显存
        torch.cuda.empty_cache()
        return super().evaluation_loop(*args, **kwargs)
    
    def prediction_step(self, model, inputs, *args, **kwargs):
        # 限制输入大小，防止显存溢出
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.numel() > 1000000:
                # 如果输入张量过大，进行截断
                if v.dim() >= 2:
                    inputs[k] = v[:, :1000000 // v.size(0)]
        
        # 在预测步骤前清理显存
        torch.cuda.empty_cache()
        return super().prediction_step(model, inputs, *args, **kwargs)
    
    def _save(self, *args, **kwargs):
        # 在保存模型前清理显存
        torch.cuda.empty_cache()
        return super()._save(*args, **kwargs)


def setup_training_args(test_mode=False, num_gpus=1, output_dir=None):
    """
    设置训练参数
    
    Args:
        test_mode: 是否为测试模式
        num_gpus: GPU数量
        output_dir: 输出目录
        
    Returns:
        training_args: 训练参数
        run_name: 运行名称
    """
    # 设置输出目录
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_test" if test_mode else datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if output_dir is None:
        output_dir = os.path.join("output", "checkpoints", run_name)
    
    # 测试模式下自定义步数和周期
    eval_steps = 50 if test_mode else 400
    save_steps = 100 if test_mode else 400
    
    # 根据GPU数量调整训练参数
    if num_gpus > 1:
        # 为多GPU环境优化参数
        per_device_batch_size = 1  # 每个GPU上的批次大小
        gradient_accumulation_steps = max(1, 4 // num_gpus)  # 根据GPU数量减少梯度累积
        dp_or_ddp = "ddp"  # 使用分布式数据并行
    else:
        # 单GPU环境
        per_device_batch_size = 1
        gradient_accumulation_steps = 1 if test_mode else 2
        dp_or_ddp = "no"
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=1,  # 固定为1，避免评估时显存溢出
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=0.1 if test_mode else 1,  # 测试模式下减少训练epoch数
        learning_rate=1e-4,
        eval_strategy="steps",  # 按步数评估
        eval_steps=eval_steps,  # 评估步数
        save_strategy="steps",
        save_steps=save_steps,  # 保存步数
        logging_dir=os.path.join("output", "logs", run_name),
        logging_steps=10,
        fp16=True,
        save_total_limit=2 if test_mode else 3,  # 测试模式下减少保存的检查点数量
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        load_best_model_at_end=True,
        metric_for_best_model="judge_score",  # 使用评测模型分数作为选择最佳模型的指标
        greater_is_better=True,  # 评测分数越高越好
        max_steps=200 if test_mode else -1,  # 测试模式下限制最大步数
        ddp_find_unused_parameters=False,  # 分布式训练优化
        dataloader_num_workers=2,  # 减少数据加载的工作线程数以节省内存
        ddp_backend="nccl" if num_gpus > 1 else None,  # 使用NCCL后端进行分布式训练
        report_to="none",  # 禁用报告以减少内存使用
    )
    
    return training_args, run_name


def save_training_results(trainer, model, final_model_dir, run_name, lora_config, training_args, 
                          train_dataset, val_dataset, baseline_metrics, final_metrics, eval_results,
                          train_data_path, val_data_path, data_path, images_dir, test_sample_ratio=0.01):
    """
    保存训练结果
    
    Args:
        trainer: 训练器
        model: 模型
        final_model_dir: 最终模型目录
        run_name: 运行名称
        lora_config: LoRA配置
        training_args: 训练参数
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        baseline_metrics: 基准评估指标
        final_metrics: 最终评估指标
        eval_results: 评估结果历史
        train_data_path: 训练数据路径
        val_data_path: 验证数据路径
        data_path: 原始数据路径
        images_dir: 图像目录
        test_sample_ratio: 测试抽样比例
    """
    # 保存模型
    trainer.save_model(final_model_dir)
    
    # 辅助函数
    def convert_to_serializable(obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        else:
            return obj
    
    # 转换配置为可序列化对象
    lora_dict = convert_to_serializable(lora_config.to_dict())
    training_dict = convert_to_serializable(training_args.to_dict())
    eval_history = convert_to_serializable(eval_results)
    
    # 创建配置字典
    config = {
        "model_name": "Qwen/Qwen2-VL-2B-Instruct",
        "lora_config": lora_dict,
        "training_args": training_dict,
        "dataset_info": {
            "total_samples": len(train_dataset) + len(val_dataset),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "train_data_path": train_data_path,
            "val_data_path": val_data_path,
            "original_data_path": data_path,
            "images_dir": images_dir,
            "sampling_rate": test_sample_ratio,
            "train_val_split": "7:1"
        },
        "metrics": {
            "baseline": baseline_metrics,
            "final": final_metrics,
            "evaluation_history": eval_history
        }
    }
    
    # 保存配置
    with open(os.path.join(final_model_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # 打印训练完成信息
    print(f"""
    训练完成!
    - 最终模型保存至: {final_model_dir}
    - 训练日志: {training_args.logging_dir}
    - 检查点: {training_args.output_dir}
    - 训练集大小: {len(train_dataset)} 样本
    - 验证集大小: {len(val_dataset)} 样本
    - 基准评估指标: {baseline_metrics}
    - 最终评估指标: {final_metrics}
    """) 