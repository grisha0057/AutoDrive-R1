import torch
import numpy as np
from transformers import TrainerCallback
from PIL import Image


def compute_metrics(eval_pred):
    """
    计算评估指标
    
    包括语言模型交叉熵损失和自定义评测模型的指标
    
    Args:
        eval_pred: 评估数据，包含logits和labels
        
    Returns:
        metrics: 指标字典
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


def evaluate_with_judge(model, eval_dataset, processor, trainer, judge_model=None, num_gpus=1, test_mode=False):
    """
    使用评测模型对验证集进行评估
    
    Args:
        model: 要评估的模型
        eval_dataset: 评估数据集
        processor: 处理器
        trainer: 训练器
        judge_model: 评测模型
        num_gpus: GPU数量
        test_mode: 是否为测试模式
        
    Returns:
        metrics: 评估指标字典
    """
    print("开始使用评测模型评估...")
    
    # 确保在评估前清理显存
    torch.cuda.empty_cache()
    
    # 将模型移到评估模式
    model.eval()
    
    # 确定设备
    if hasattr(model, "device_map") and model.device_map:
        # 如果模型使用device_map，选择第一个设备用于评估
        device = next(model.parameters()).device
    else:
        # 否则使用cuda:0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 初始化评测模型（如果未提供）
    if judge_model is None:
        try:
            # 更改评测模型到cuda:1，减少cuda:0的显存压力
            judge_device = torch.device("cuda:1" if num_gpus > 1 else "cuda:0")
            from ..models.judge import SceneryJudge
            judge_model = SceneryJudge().to(judge_device)
            print(f"成功初始化评测模型，使用设备: {judge_device}")
        except Exception as e:
            print(f"初始化评测模型失败: {str(e)}")
            return {"judge_score": 0.0}
    
    # 生成预测结果 - 修改为小批量处理以减少峰值内存使用
    print("生成预测结果...")
    questions = []
    references = []
    predictions = []
    
    # 小批量处理，以减少峰值内存使用
    batch_size = 1  # 每次只处理一个样本
    for i in range(0, len(eval_dataset), batch_size):
        # 在每个批次处理之间清理缓存
        torch.cuda.empty_cache()
        
        batch_end = min(i + batch_size, len(eval_dataset))
        batch_questions, batch_references, batch_predictions = [], [], []
        
        for j in range(i, batch_end):
            item = eval_dataset.data[j]
            image_path = item["image_path"]
            question = item["question"]
            reference = item["answer"]
            
            try:
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
                
                # 减小处理的图像尺寸以节省显存
                inputs = processor.apply_chat_template(
                    conversation,
                    return_tensors="pt",
                    tokenize=True
                )
                
                print(f"DEBUG: inputs类型: {type(inputs)}")
                if hasattr(inputs, "items"):
                    print(f"DEBUG: inputs包含keys: {list(inputs.keys())}")
                
                # 确定使用的设备
                input_device = "cuda:1" if num_gpus > 1 else device  # 如果有多个GPU，使用第二个GPU
                
                # 根据返回类型处理移动到设备
                if isinstance(inputs, dict) or hasattr(inputs, "items"):
                    # 处理字典类型和BatchFeature类型
                    inputs = {k: v.to(input_device) for k, v in inputs.items()}
                elif isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(input_device)
                else:
                    print(f"错误: 无法处理的inputs类型: {type(inputs)}")
                    batch_questions.append(question)
                    batch_references.append([reference])
                    batch_predictions.append("处理错误")
                    continue
                
                # 限制生成的token数量，减少显存使用
                max_tokens = 32 if test_mode else 128
                
                with torch.inference_mode():
                    # 分批处理输入以减少显存峰值
                    if hasattr(model, "generate"):
                        # 根据inputs类型调用generate
                        if isinstance(inputs, dict) or hasattr(inputs, "items"):
                            # 处理字典类型和BatchFeature类型
                            output = model.generate(
                                **inputs,
                                max_new_tokens=max_tokens,
                                do_sample=False
                            )
                        elif isinstance(inputs, torch.Tensor):
                            # 如果是张量，直接作为input_ids传入
                            output = model.generate(
                                inputs,
                                max_new_tokens=max_tokens,
                                do_sample=False
                            )
                        else:
                            print(f"错误: generate无法处理的inputs类型: {type(inputs)}")
                            output = torch.ones((1, 10), dtype=torch.long, device=input_device) * processor.tokenizer.eos_token_id
                    else:
                        # 如果模型没有generate方法，回退到基本输出
                        output = torch.ones((1, 10), dtype=torch.long, device=input_device) * processor.tokenizer.eos_token_id
                
                generated_text = processor.tokenizer.decode(output[0], skip_special_tokens=True)
                prediction = generated_text.split("assistant:")[-1].strip()
                
                batch_questions.append(question)
                batch_references.append([reference])
                batch_predictions.append(prediction)
            except Exception as e:
                print(f"生成预测时出错 (idx={j}, image={image_path}): {str(e)}")
                # 在出错情况下添加空值，保持索引一致
                batch_questions.append(question)
                batch_references.append([reference])
                batch_predictions.append("")
                
            # 每处理一个样本后清理缓存
            torch.cuda.empty_cache()
        
        # 将本批次的结果添加到总结果中
        questions.extend(batch_questions)
        references.extend(batch_references)
        predictions.extend(batch_predictions)
        
        print(f"已处理 {batch_end}/{len(eval_dataset)} 个样本")
    
    # 使用评测模型评估
    try:
        print("使用评测模型计算分数...")
        torch.cuda.empty_cache()  # 确保评测前清理显存
        
        # 小批量评测以减少显存使用
        all_scores = []
        batch_size = 3  # 每批评测几个样本
        
        for i in range(0, len(questions), batch_size):
            batch_end = min(i + batch_size, len(questions))
            batch_questions = questions[i:batch_end]
            batch_references = references[i:batch_end]
            batch_predictions = predictions[i:batch_end]
            
            batch_scores = judge_model.compute(batch_questions, batch_references, batch_predictions)
            all_scores.extend(batch_scores.tolist())
            
            torch.cuda.empty_cache()  # 每批评测后清理显存
        
        judge_score = sum(all_scores) / len(all_scores) if all_scores else 0
        print(f"评测模型评分: {judge_score:.4f}")
        
        # 返回指标结果
        return {
            "judge_score": judge_score,
            "judge_scores": all_scores
        }
    except Exception as e:
        print(f"评测模型评估失败: {str(e)}")
        return {"judge_score": 0.0}


class EvaluationCallback(TrainerCallback):
    """
    自定义评估回调类
    """
    def __init__(self, trainer, eval_dataset, processor, judge_model=None, test_mode=False, num_gpus=1):
        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.judge_model = judge_model
        self.best_judge_score = 0.0
        self.eval_results = []
        self.test_mode = test_mode
        self.num_gpus = num_gpus
    
    def on_step_end(self, args, state, control, **kwargs):
        # 测试模式下减少评估频率，每100步评估一次
        eval_frequency = 100 if self.test_mode else 400
        
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
                self.judge_model,
                num_gpus=self.num_gpus,
                test_mode=self.test_mode
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


def calculate_model_perplexity(model, dataset, processor, batch_size=1, max_samples=None):
    """
    计算模型在给定数据集上的perplexity (困惑度) loss
    
    Args:
        model: 要评估的模型
        dataset: 评估数据集
        processor: 文本处理器
        batch_size: 批处理大小
        max_samples: 最大样本数，如果为None则使用全部数据集
    
    Returns:
        avg_loss: 平均loss值
    """
    print("开始计算模型困惑度...")
    model.eval()  # 设为评估模式
    
    total_loss = 0.0
    sample_count = 0
    max_count = max_samples if max_samples is not None else len(dataset)
    
    # 遍历数据集的每个样本
    for i in range(0, min(len(dataset), max_count), batch_size):
        torch.cuda.empty_cache()  # 清理GPU缓存
        
        batch_loss = 0.0
        batch_end = min(i + batch_size, min(len(dataset), max_count))
        
        for j in range(i, batch_end):
            try:
                # 获取样本
                sample = dataset[j]
                
                # 确保输入在正确的设备上
                sample_device = next(model.parameters()).device
                inputs = {k: v.unsqueeze(0).to(sample_device) for k, v in sample.items() 
                          if isinstance(v, torch.Tensor)}
                
                # 确保包含labels
                if "labels" not in inputs:
                    print(f"警告: 样本 {j} 不包含labels字段，跳过")
                    continue
                
                # 前向传播计算loss
                with torch.no_grad():
                    outputs = model(**inputs)
                    loss = outputs.loss
                
                batch_loss += loss.item()
                sample_count += 1
                
                # 打印进度
                if sample_count % 5 == 0:
                    print(f"已处理 {sample_count}/{max_count} 个样本, 当前平均loss: {total_loss/sample_count:.4f}")
                
            except Exception as e:
                print(f"处理样本 {j} 时出错: {str(e)}")
                continue
        
        # 累加批次loss
        total_loss += batch_loss
    
    # 计算平均loss
    avg_loss = total_loss / sample_count if sample_count > 0 else float('nan')
    print(f"完成困惑度计算, 平均loss: {avg_loss:.4f}, 处理样本数: {sample_count}")
    
    return avg_loss 