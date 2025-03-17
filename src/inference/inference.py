import os
import torch
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

# 示例使用
if __name__ == "__main__":
    # 使用找到的实际图像路径
    test_image_path = "/cloud/data/scenery/images/train/1cff1954a28837441e7376a1a168186e/0.jpg"
    test_question = "What can you see in this image?"
    
    if os.path.exists(test_image_path):
        answer = generate_answer(test_image_path, test_question)
        print(f"Question: {test_question}")
        print(f"Answer: {answer}")
    else:
        print(f"Test image not found: {test_image_path}")
        print("Please update the test_image_path with a valid image path.") 