import os
import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel, PeftConfig

# 模型路径配置
base_model_name = "Qwen/Qwen2-VL-2B-Instruct"  # 基础模型
lora_model_path = "output/final_model/20250317_234018"  # 实际模型路径

# 加载处理器
processor = AutoProcessor.from_pretrained(base_model_name)

# 加载基础模型
print("Loading base model...")
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 加载LoRA配置
peft_config = PeftConfig.from_pretrained(lora_model_path)
print(f"Loading LoRA adapter from {lora_model_path}...")

# 加载LoRA模型
model = PeftModel.from_pretrained(base_model, lora_model_path)

# 设置为评估模式
model.eval()

def generate_answer(image_path, question):
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
            max_new_tokens=256,
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

# 示例使用
if __name__ == "__main__":
    # 从parquet文件加载数据
    print("正在加载数据集...")
    try:
        df = pd.read_parquet("/cloud/data/scenery/train.parquet")
        print(f"成功加载数据集，共有{len(df)}条记录")
    except Exception as e:
        print(f"加载数据集时出错: {str(e)}")
        # 尝试使用备用路径
        try:
            df = pd.read_parquet("/cloud/scenery/train.parquet")
            print(f"成功从备用路径加载数据集，共有{len(df)}条记录")
        except Exception as e2:
            print(f"从备用路径加载数据集时出错: {str(e2)}")
            exit(1)
    
    # 选择几个样本进行测试
    # 随机选择5个不同的segment_id
    test_segments = df['segment_id'].sample(5).tolist()
    print(f"选择的测试segment_id: {test_segments}")
    
    # 对每个segment进行测试
    for segment_id in test_segments:
        # 获取该segment的问题和答案
        segment_data = df[df['segment_id'] == segment_id].iloc[0]
        question = segment_data.get('question', "What can you see in this image?")
        ground_truth = segment_data.get('answer', "No answer provided in dataset")
        
        print(f"\n{'='*50}")
        print(f"测试Segment ID: {segment_id}")
        print(f"问题: {question}")
        print(f"参考答案: {ground_truth}")
        print(f"{'='*50}")
        
        # 获取该segment下的所有图片
        base_image_path = f"/cloud/data/scenery/images/train/{segment_id}"
        if not os.path.exists(base_image_path):
            base_image_path = f"/cloud/scenery/images/train/{segment_id}"
            if not os.path.exists(base_image_path):
                print(f"图像路径不存在: {base_image_path}")
                continue
        
        # 获取该文件夹下的所有jpg图片
        image_files = [f for f in os.listdir(base_image_path) if f.endswith('.jpg')]
        
        for img_file in image_files:
            img_path = os.path.join(base_image_path, img_file)
            print(f"\n--- 图像: {img_path} ---")
            
            try:
                # 获取模型推理结果
                model_answer = generate_answer(img_path, question)
                print(f"模型回答: {model_answer}")
            except Exception as e:
                print(f"处理图像时出错: {str(e)}")
    
    print("\n推理完成!") 