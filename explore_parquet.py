import pandas as pd
import matplotlib.pyplot as plt

# 读取 parquet 文件
df = pd.read_parquet('data/scenery/train.parquet')

# 显示基本信息
print("\n=== 数据基本信息 ===")
print(f"数据形状: {df.shape}")
print("\n=== 列名称 ===")
print(df.columns.tolist())
print("\n=== 数据类型 ===")
print(df.dtypes)
print("\n=== 数据预览 ===")
print(df.head())
print("\n=== 基本统计信息 ===")

# 检查是否有缺失值
print("\n=== 缺失值统计 ===")
print(df.isnull().sum()) 