# AutoDrive-R1

## 环境设置

使用 Conda 创建虚拟环境并安装依赖：

```bash
# 创建 conda 环境
conda create -n autodrive python=3.10 -y

# 激活环境
conda activate autodrive

# 安装依赖
pip install -r requirements.txt

# 安装nuScenes开发工具包（如果需要使用nuScenes数据集）
pip install nuscenes-devkit
```

## 使用方法

### 使用自定义图像

1. 将测试图像放入 `test_images` 目录
2. 运行推理脚本：

```bash
# 激活环境（如果尚未激活）
conda activate autodrive

# 运行脚本
python infer.py --image test_images/your_image.jpg --output output.jpg
```

### 使用nuScenes mini数据集

1. 下载并解压nuScenes mini数据集（约4GB）：

```bash
# 创建数据目录
mkdir -p data

# 下载数据集（可能需要一些时间）
cd data
curl -L -o v1.0-mini.tgz https://www.nuscenes.org/data/v1.0-mini.tgz

# 解压数据集
tar -xzf v1.0-mini.tgz
cd ..
```

2. 使用nuScenes数据集运行脚本：

```bash
# 随机选择一个样本
python infer.py --nuscenes --data_root data --output output.jpg --simulate

# 指定样本索引
python infer.py --nuscenes --data_root data --sample_idx 10 --output output.jpg --simulate
```

注意：首次运行时，脚本会自动从 ModelScope 或 Hugging Face 下载 Qwen2.5-VL-7B 模型。如果不想下载大模型，可以使用 `--simulate` 参数启用模拟模式。

## 功能

该脚本使用 Qwen2.5-VL-7B 多模态大语言模型来：
1. 分析驾驶场景
2. 预测未来的速度和曲率变化
3. 可视化预测结果

## 硬件要求

- 至少 16GB 内存
- 支持 CUDA 的 GPU（推荐至少 8GB 显存）
- 约 15GB 磁盘空间（用于存储模型）
- 额外 4GB 磁盘空间（如果使用 nuScenes mini 数据集）

## 自定义模型

使用其他模型，可以通过 `--model` 参数指定：

```bash
python infer.py --image test_images/your_image.jpg --model llava --output output.jpg
```

目前支持的模型类型：
- qwen (默认): Qwen2.5-VL-7B 模型