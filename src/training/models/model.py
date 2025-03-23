import os
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model


def setup_model(model_name="Qwen/Qwen2-VL-2B-Instruct"):
    """
    设置并初始化基础模型和处理器
    
    Args:
        model_name: 预训练模型名称
        
    Returns:
        model: 初始化好的模型
        processor: 模型处理器
    """
    # 检查可用的GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"可用GPU数量: {num_gpus}")

    # 设置设备映射策略，使用所有可用GPU
    if num_gpus > 1:
        device_map = "auto"
        # 配置环境变量以允许PyTorch更好地管理显存
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    else:
        device_map = "auto"

    # 添加显存优化设置
    torch.cuda.empty_cache()  # 清空缓存
    # 设置较小的限制，以避免在Attention计算中分配过大的内存
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8"

    # 加载处理器
    processor = AutoProcessor.from_pretrained(model_name)
    
    # 加载模型
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map=device_map,  # 自动将模型分布到多个GPU上
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        ),
        torch_dtype=torch.float16,
    )

    # 更严格地控制缓存以防止显存泄漏
    model.config.use_cache = False
    
    return model, processor


def prepare_for_training(model):
    """
    为训练准备模型（添加LoRA适配器）
    
    Args:
        model: 基础模型
        
    Returns:
        model: 添加了LoRA适配器的模型
    """
    model.enable_input_require_grads()
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["vision_tower", "vision_proj"],
    )

    # 应用LoRA配置
    model = get_peft_model(model, lora_config)

    # 打印可训练参数信息
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"可训练参数: {trainable_params} || 总参数: {all_param} || 可训练比例: {100 * trainable_params / all_param}"
    )

    # 确保LoRA参数可训练
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
            
    return model, lora_config


def generate_predictions(model, eval_dataset, processor, device="cuda"):
    """
    使用模型为验证集生成预测结果
    
    Args:
        model: 训练好的模型
        eval_dataset: 评估数据集
        processor: 模型处理器
        device: 使用的设备
        
    Returns:
        questions: 问题列表
        references: 参考答案列表
        predictions: 预测结果列表
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
                tokenize=True
            )
            
            print(f"DEBUG: inputs类型: {type(inputs)}")
            if hasattr(inputs, "items"):
                print(f"DEBUG: inputs包含keys: {list(inputs.keys())}")
            
            # 根据返回类型处理移动到设备
            target_device = "cuda:0" if isinstance(device, str) and device == "cuda" else device
            
            if isinstance(inputs, dict) or hasattr(inputs, "items"):
                # 处理字典类型和BatchFeature类型
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
            elif isinstance(inputs, torch.Tensor):
                inputs = inputs.to(target_device)
            else:
                print(f"错误: 无法处理的inputs类型: {type(inputs)}")
                predictions.append("")
                questions.append(question)
                references.append([reference])
                continue
            
            with torch.inference_mode():
                # 分批处理输入以减少显存峰值
                if hasattr(model, "generate"):
                    # 根据inputs类型调用generate
                    if isinstance(inputs, dict) or hasattr(inputs, "items"):
                        # 处理字典类型和BatchFeature类型
                        output = model.generate(
                            **inputs,
                            max_new_tokens=256,
                            do_sample=False
                        )
                    elif isinstance(inputs, torch.Tensor):
                        # 如果是张量，直接作为input_ids传入
                        output = model.generate(
                            inputs,
                            max_new_tokens=256,
                            do_sample=False
                        )
                    else:
                        print(f"错误: generate无法处理的inputs类型: {type(inputs)}")
                        output = torch.ones((1, 10), dtype=torch.long, device=target_device) * processor.tokenizer.eos_token_id
                else:
                    # 如果模型没有generate方法，回退到基本输出
                    output = torch.ones((1, 10), dtype=torch.long, device=target_device) * processor.tokenizer.eos_token_id
            
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