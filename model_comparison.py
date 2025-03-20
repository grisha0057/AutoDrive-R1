import os
import torch
import json
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, pipeline
from peft import PeftModel, PeftConfig
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

# 模型路径配置
base_model_name = "Qwen/Qwen2-VL-2B-Instruct"  # 基础模型
lora_model_path = "output/final_model/20250317_234018"  # 微调后的模型路径

# 加载处理器
print("加载处理器...")
processor = AutoProcessor.from_pretrained(base_model_name)

# 加载原始基础模型
print("加载原始模型...")
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
base_model.eval()

# 加载SFT微调后的模型
print(f"加载微调后的模型从 {lora_model_path}...")
try:
    # 创建一个新的基础模型实例，避免共享权重
    sft_base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    # 加载LoRA配置
    peft_config = PeftConfig.from_pretrained(lora_model_path)
    # 加载LoRA模型
    sft_model = PeftModel.from_pretrained(sft_base_model, lora_model_path)
    sft_model.eval()
    models_loaded = True
except Exception as e:
    print(f"加载微调模型失败: {str(e)}")
    models_loaded = False

# 加载Lingo-Judge评分模型
print("加载Lingo-Judge评分模型...")
judge_pipe = None
try:
    judge_model_name = 'wayveai/Lingo-Judge'
    judge_pipe = pipeline("text-classification", model=judge_model_name)
    judge_loaded = True
except Exception as e:
    print(f"加载评分模型失败: {str(e)}")
    judge_loaded = False

def generate_answer(model, image_path, question):
    """使用模型生成对图像问题的回答"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # 构建对话
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"Please answer the question about this scene: {question}"}
            ]
        }
    ]
    
    # 处理输入
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    # 解码输出
    response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取助手的回答部分
    assistant_prefix = "<|im_start|>assistant\n"
    if assistant_prefix in response:
        response = response.split(assistant_prefix)[-1].split("<|im_end|>")[0].strip()
    
    return response

def judge_answer(question, reference_answer, prediction):
    """使用Lingo-Judge评分模型评估答案质量"""
    global judge_pipe
    
    if not judge_loaded or judge_pipe is None:
        return 0.0
    
    input_text = f"[CLS]\nQuestion: {question}\nAnswer: {reference_answer}\nStudent: {prediction}"
    result = judge_pipe(input_text)
    score = result[0]['score']
    return score

def batch_judge_answers(questions, references, predictions, batch_size=16):
    """批量评估答案质量"""
    if not judge_loaded or judge_pipe is None:
        return [0.0] * len(questions)
    
    input_texts = [
        f"[CLS]\nQuestion: {q}\nAnswer: {r}\nStudent: {p}" 
        for q, r, p in zip(questions, references, predictions)
    ]
    
    all_scores = []
    for i in range(0, len(input_texts), batch_size):
        batch = input_texts[i:i+batch_size]
        results = judge_pipe(batch)
        scores = [r['score'] for r in results]
        all_scores.extend(scores)
    
    return all_scores

def load_test_data_from_parquet(parquet_path, images_dir, num_samples=100):
    """从train.parquet文件加载真实问题和答案数据，并构建正确的图片路径"""
    print(f"从parquet文件加载数据: {parquet_path}")
    
    if not os.path.exists(parquet_path):
        raise ValueError(f"Parquet文件不存在: {parquet_path}")
    
    if not os.path.exists(images_dir):
        raise ValueError(f"图像目录不存在: {images_dir}")
    
    try:
        # 加载parquet文件
        df = pd.read_parquet(parquet_path)
        print(f"成功加载parquet文件，共有{len(df)}条记录")
        
        # 确保必要的列存在
        required_columns = ['segment_id', 'question', 'answer']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Parquet文件缺少必要的列: {missing_columns}")
        
        # 随机选择指定数量的样本
        if len(df) > num_samples:
            df = df.sample(n=num_samples, random_state=42)
            print(f"随机选择了{len(df)}条记录进行测试")
        else:
            print(f"Parquet文件只有{len(df)}条记录，将全部使用")
        
        # 构建数据集
        test_data = []
        valid_count = 0
        
        for _, row in df.iterrows():
            segment_id = row['segment_id']
            image_path = os.path.join(images_dir, segment_id, "0.jpg")
            
            # 检查图像文件是否存在
            if os.path.exists(image_path):
                test_data.append({
                    "image_path": image_path,
                    "question": row['question'],
                    "reference": row['answer']
                })
                valid_count += 1
            else:
                print(f"警告: 图像不存在: {image_path}")
        
        if not test_data:
            raise ValueError(f"无法找到任何与segment_id对应的图像文件")
        
        print(f"成功构建数据集，包含{len(test_data)}个有效样本")
        return Dataset.from_list(test_data)
    
    except Exception as e:
        print(f"加载Parquet文件时出错: {str(e)}")
        raise

def process_batch_for_model(model, batch, batch_size=4):
    """批量处理模型预测"""
    predictions = []
    for i in range(0, len(batch['image_path']), batch_size):
        mini_batch_paths = batch['image_path'][i:i+batch_size]
        mini_batch_questions = batch['question'][i:i+batch_size]
        
        mini_batch_preds = []
        # 这里仍然是顺序处理单个样本，但每个mini_batch释放一次GPU内存
        for img_path, question in zip(mini_batch_paths, mini_batch_questions):
            try:
                pred = generate_answer(model, img_path, question)
                mini_batch_preds.append(pred)
            except Exception as e:
                print(f"处理图像时出错: {img_path}, 错误: {str(e)}")
                mini_batch_preds.append("Error processing image")
        
        predictions.extend(mini_batch_preds)
    return predictions

def compare_models_on_dataset(dataset, batch_size=16):
    """在数据集上比较两个模型的性能"""
    results = {
        "原始模型分数": [],
        "微调模型分数": [],
        "问题": [],
        "参考答案": [],
        "原始模型答案": [],
        "微调模型答案": []
    }
    
    # 加载数据集
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="处理批次")):
        print(f"处理批次 {batch_idx+1}/{len(dataloader)}")
        
        # 从原始模型获取预测
        print("获取原始模型预测...")
        base_predictions = process_batch_for_model(base_model, batch)
        
        # 如果微调模型已加载，从微调模型获取预测
        sft_predictions = []
        if models_loaded:
            print("获取微调模型预测...")
            sft_predictions = process_batch_for_model(sft_model, batch)
        
        # 批量评估答案
        print("评估原始模型分数...")
        base_scores = batch_judge_answers(batch['question'], batch['reference'], base_predictions)
        results["原始模型分数"].extend(base_scores)
        results["问题"].extend(batch['question'])
        results["参考答案"].extend(batch['reference'])
        results["原始模型答案"].extend(base_predictions)
        
        if models_loaded and sft_predictions:
            print("评估微调模型分数...")
            sft_scores = batch_judge_answers(batch['question'], batch['reference'], sft_predictions)
            results["微调模型分数"].extend(sft_scores)
            results["微调模型答案"].extend(sft_predictions)
        
        # 打印每个批次的平均分数
        avg_base_score = sum(base_scores) / len(base_scores)
        print(f"批次 {batch_idx+1} 原始模型平均分数: {avg_base_score:.4f}")
        
        if models_loaded and sft_predictions:
            avg_sft_score = sum(sft_scores) / len(sft_scores)
            print(f"批次 {batch_idx+1} 微调模型平均分数: {avg_sft_score:.4f}")
    
    return results

# 主函数
if __name__ == "__main__":
    # 图像目录和Parquet文件路径
    images_dir = "/cloud/data/scenery/images/train"  # 训练图像目录
    parquet_path = "/cloud/data/scenery/train.parquet"  # 训练数据parquet文件路径
    
    try:
        # 准备测试数据集
        num_samples = 100
        print(f"准备{num_samples}个测试样本...")
        test_dataset = load_test_data_from_parquet(parquet_path, images_dir, num_samples=num_samples)
        print(f"成功加载数据集，共{len(test_dataset)}个样本")
        
        # 在数据集上运行模型比较
        results = compare_models_on_dataset(test_dataset)
        
        # 计算平均分数
        base_avg = sum(results["原始模型分数"]) / len(results["原始模型分数"])
        
        # 输出总结结果
        print("\n" + "="*50)
        print("测试结果总结:")
        print(f"测试样本数量: {len(results['原始模型分数'])}")
        print(f"原始模型平均分数: {base_avg:.4f}")
        
        if models_loaded and results["微调模型分数"]:
            sft_avg = sum(results["微调模型分数"]) / len(results["微调模型分数"])
            print(f"微调模型平均分数: {sft_avg:.4f}")
            print(f"分数变化: {sft_avg - base_avg:.4f}")
        
        print("="*50)
        
        # 保存详细结果到JSON文件
        with open("model_comparison_results.json", "w") as f:
            json.dump({
                "base_scores": results["原始模型分数"],
                "sft_scores": results["微调模型分数"] if models_loaded else [],
                "base_avg": base_avg,
                "sft_avg": sft_avg if models_loaded else 0.0,
                "improvement": (sft_avg - base_avg) if models_loaded else 0.0,
                "num_samples": len(results["原始模型分数"]),
                "samples": [
                    {
                        "question": q,
                        "reference": r,
                        "base_answer": ba,
                        "sft_answer": sa if models_loaded and results["微调模型答案"] else "",
                        "base_score": bs,
                        "sft_score": ss if models_loaded and results["微调模型分数"] else 0.0
                    }
                    for q, r, ba, sa, bs, ss in zip(
                        results["问题"], 
                        results["参考答案"], 
                        results["原始模型答案"], 
                        results["微调模型答案"] if models_loaded and results["微调模型答案"] else [""] * len(results["问题"]),
                        results["原始模型分数"],
                        results["微调模型分数"] if models_loaded and results["微调模型分数"] else [0.0] * len(results["问题"])
                    )
                ]
            }, f, indent=2)
        
        print("详细结果已保存到 model_comparison_results.json")
        
        # 保存CSV格式的详细比较结果
        try:
            comparison_df = pd.DataFrame({
                "问题": results["问题"],
                "参考答案": results["参考答案"],
                "原始模型答案": results["原始模型答案"],
                "原始模型分数": results["原始模型分数"],
                "微调模型答案": results["微调模型答案"] if models_loaded and results["微调模型答案"] else [""] * len(results["问题"]),
                "微调模型分数": results["微调模型分数"] if models_loaded and results["微调模型分数"] else [0.0] * len(results["问题"]),
                "分数提升": [sft - base for sft, base in zip(
                    results["微调模型分数"] if models_loaded and results["微调模型分数"] else [0.0] * len(results["问题"]),
                    results["原始模型分数"]
                )]
            })
            comparison_df.to_csv("model_comparison_results.csv", index=False)
            print("详细结果也已保存为CSV格式: model_comparison_results.csv")
        except Exception as e:
            print(f"保存CSV文件时出错: {str(e)}")
        
    except Exception as e:
        print(f"运行测试时出错: {str(e)}")
        import traceback
        traceback.print_exc()