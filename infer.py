import os
import argparse
import cv2
import numpy as np
import torch
from PIL import Image
import base64
import re
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

# 加载Qwen2.5-VL-7B模型
def load_qwen_model():
    """加载Qwen2.5-VL-3B模型"""
    print("正在加载Qwen2.5-VL-3B模型...")
    
    # 直接从Hugging Face下载模型
    model_dir = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    try:
        # 使用AutoModelForVision2Seq而不是AutoModelForCausalLM
        from transformers import AutoModelForVision2Seq
        
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_dir, 
            device_map="auto",
            trust_remote_code=True
        ).eval()
        return model, tokenizer
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("将使用模拟模式运行")
        return None, None

# 加载nuScenes数据集
def load_nuscenes_sample(data_root, sample_idx=None):
    """加载nuScenes数据集中的样本图像"""
    print("正在加载nuScenes数据集...")
    
    # 加载sample.json文件
    sample_file = os.path.join(data_root, "v1.0-mini", "sample.json")
    if not os.path.exists(sample_file):
        print(f"找不到sample.json文件: {sample_file}")
        return None
    
    with open(sample_file, 'r') as f:
        samples = json.load(f)
    
    # 如果没有指定样本索引，随机选择一个
    if sample_idx is None or sample_idx >= len(samples):
        import random
        sample_idx = random.randint(0, len(samples)-1)
        print(f"随机选择样本索引: {sample_idx}")
    
    sample = samples[sample_idx]
    print(f"样本token: {sample['token']}")
    
    # 加载sample_data.json文件以获取图像路径
    sample_data_file = os.path.join(data_root, "v1.0-mini", "sample_data.json")
    if not os.path.exists(sample_data_file):
        print(f"找不到sample_data.json文件: {sample_data_file}")
        return None
    
    with open(sample_data_file, 'r') as f:
        sample_data_list = json.load(f)
    
    # 查找前置摄像头图像
    cam_front_data = None
    for data in sample_data_list:
        if data['sample_token'] == sample['token'] and 'CAM_FRONT' in data.get('filename', ''):
            cam_front_data = data
            break
    
    if cam_front_data is None:
        print("找不到前置摄像头图像")
        return None
    
    # 构建图像路径
    image_path = os.path.join(data_root, cam_front_data['filename'])
    if not os.path.exists(image_path):
        print(f"找不到图像文件: {image_path}")
        return None
    
    print(f"加载图像: {image_path}")
    return image_path

# 简化的场景描述函数
def scene_description(image_path, model_type="qwen", model=None, tokenizer=None, simulate=False):
    """分析驾驶场景并返回描述"""
    if simulate or model is None or tokenizer is None:
        print("使用模拟模式生成场景描述")
        return "这是一个城市道路场景。道路上有多辆车辆行驶，前方有一个十字路口，交通灯显示绿色。道路两侧有行人和建筑物。车道标记清晰可见，包括实线和虚线。"
    
    if model_type == "qwen":
        # 加载图像
        image = Image.open(image_path)
        
        # 准备提示
        prompt = "描述这个驾驶场景，包括交通灯、其他车辆和行人的移动以及车道标记。"
        
        # 使用Qwen2.5-VL-7B模型进行推理
        query = tokenizer.from_list_format([
            {'image': image},
            {'text': prompt}
        ])
        response, _ = model.chat(tokenizer, query=query, history=None)
        
        return response
    else:
        # 这里可以添加其他模型的实现
        return "使用非Qwen模型时的占位描述"

# 简化的运动预测函数
def predict_motion(image_path, scene_desc, model_type="qwen", model=None, tokenizer=None, simulate=False):
    """预测未来的速度和曲率"""
    if simulate or model is None or tokenizer is None:
        print("使用模拟模式生成运动预测")
        return "[[20,0], [20,0], [19,0.01], [19,0.01], [18,0.02], [18,0.02], [17,0.01], [17,0], [17,0], [17,0]]"
    
    if model_type == "qwen":
        # 加载图像
        image = Image.open(image_path)
        
        # 准备提示
        prompt = f"""
        基于这个驾驶场景图像和以下描述：
        {scene_desc}
        
        预测未来5秒的速度和曲率变化。使用格式[速度,曲率]表示，
        例如[20,0.05]表示速度20m/s，向左转弯曲率0.05。
        负曲率表示向右转弯。请提供10个预测点（每0.5秒一个）。
        只需要返回预测结果，格式为[[速度1,曲率1], [速度2,曲率2], ...]
        """
        
        # 使用Qwen2.5-VL-7B模型进行推理
        query = tokenizer.from_list_format([
            {'image': image},
            {'text': prompt}
        ])
        response, _ = model.chat(tokenizer, query=query, history=None)
        
        # 确保返回格式正确
        if not re.search(r"\[\[.*\]\]", response):
            print("警告: 模型返回格式不正确，使用默认预测")
            return "[[20,0], [20,0], [19,0.01], [19,0.01], [18,0.02], [18,0.02], [17,0.01], [17,0], [17,0], [17,0]]"
        
        return response
    else:
        # 这里可以添加其他模型的实现
        return "[[20,0], [20,0], [19,0.01], [19,0.01], [18,0.02], [18,0.02], [17,0.01], [17,0], [17,0], [17,0]]"

# 简化的可视化函数
def visualize_prediction(image_path, prediction, output_path):
    """在图像上可视化预测轨迹"""
    # 解析预测结果
    import re
    coordinates = re.findall(r"\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]", prediction)
    if not coordinates:
        print("无法解析预测结果")
        return
    
    # 加载图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法加载图像: {image_path}")
        return
    
    # 简化的轨迹绘制（仅作示例）
    h, w = img.shape[:2]
    start_x, start_y = w // 2, h - 50  # 假设起点在图像底部中央
    
    for i, (speed, curve) in enumerate(coordinates):
        speed, curve = float(speed), float(curve)
        # 简化的轨迹计算
        next_x = start_x + i * 20 * (1 if curve >= 0 else -1)
        next_y = start_y - i * 10
        
        # 确保点在图像范围内
        next_x = max(0, min(w-1, next_x))
        next_y = max(0, min(h-1, next_y))
        
        # 绘制点和线
        cv2.circle(img, (int(next_x), int(next_y)), 5, (0, 0, 255), -1)
        if i > 0:
            prev_x = start_x + (i-1) * 20 * (1 if float(coordinates[i-1][1]) >= 0 else -1)
            prev_y = start_y - (i-1) * 10
            prev_x = max(0, min(w-1, prev_x))
            prev_y = max(0, min(h-1, prev_y))
            cv2.line(img, (int(prev_x), int(prev_y)), (int(next_x), int(next_y)), (0, 255, 0), 2)
    
    # 保存结果
    cv2.imwrite(output_path, img)
    print(f"可视化结果已保存到: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="简化版自动驾驶测试脚本")
    parser.add_argument("--image", type=str, help="输入图像路径")
    parser.add_argument("--model", type=str, default="qwen", help="使用的模型类型: qwen, llava, llama")
    parser.add_argument("--output", type=str, default="outputs/output.jpg", help="输出图像路径")
    parser.add_argument("--simulate", action="store_true", help="使用模拟模式，不加载大型模型")
    parser.add_argument("--nuscenes", action="store_true", help="使用nuScenes数据集")
    parser.add_argument("--data_root", type=str, default="data", help="nuScenes数据集根目录")
    parser.add_argument("--sample_idx", type=int, help="nuScenes样本索引")
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建输出目录: {output_dir}")
    
    # 如果没有指定输出文件名，添加时间戳
    if args.output == "outputs/output.jpg":
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.nuscenes:
            sample_info = f"_sample{args.sample_idx}" if args.sample_idx is not None else ""
            args.output = f"outputs/nuscenes{sample_info}_{timestamp}.jpg"
        else:
            image_name = os.path.basename(args.image).split('.')[0] if args.image else "unknown"
            args.output = f"outputs/{image_name}_{timestamp}.jpg"
        print(f"输出文件: {args.output}")
    
    # 确定输入图像路径
    image_path = args.image
    if args.nuscenes:
        image_path = load_nuscenes_sample(args.data_root, args.sample_idx)
        if image_path is None:
            print("无法加载nuScenes样本图像，退出")
            return
    elif image_path is None:
        print("请提供输入图像路径或使用--nuscenes选项")
        return
    
    # 加载模型（如果使用Qwen且不是模拟模式）
    model, tokenizer = None, None
    if args.model == "qwen" and not args.simulate:
        model, tokenizer = load_qwen_model()
    
    # 1. 场景描述
    print("正在分析场景...")
    scene_desc = scene_description(image_path, args.model, model, tokenizer, args.simulate)
    print(f"场景描述: {scene_desc}")
    
    # 2. 预测运动
    print("正在预测运动...")
    motion_pred = predict_motion(image_path, scene_desc, args.model, model, tokenizer, args.simulate)
    print(f"运动预测: {motion_pred}")
    
    # 3. 可视化结果
    print("正在可视化结果...")
    visualize_prediction(image_path, motion_pred, args.output)
    
    print("处理完成!")

if __name__ == "__main__":
    main()