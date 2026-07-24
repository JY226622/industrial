"""
工业互联网大作业 - 步骤1：真实数据集加载
使用 UCI AI4I 2020 预测性维护数据集（真实工业场景标准数据集）
数据源：https://archive.ics.uci.edu/dataset/601/ai4i
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import urllib.request

# 数据集下载地址
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"

def get_data_dir():
    return os.path.join(os.path.dirname(__file__), '..', 'data')

def download_dataset():
    """下载AI4I 2020数据集"""
    data_dir = get_data_dir()
    filepath = os.path.join(data_dir, 'ai4i2020_raw.csv')
    
    if os.path.exists(filepath):
        print(f"数据集已存在: {filepath}")
        return filepath
    
    print(f"正在从UCI下载AI4I 2020数据集...")
    print(f"下载地址: {DATASET_URL}")
    
    try:
        urllib.request.urlretrieve(DATASET_URL, filepath)
        print(f"下载完成: {filepath}")
        return filepath
    except Exception as e:
        print(f"下载失败: {e}")
        print("将尝试生成备用数据...")
        return None

def load_raw_data(filepath):
    """加载原始数据"""
    df = pd.read_csv(filepath)
    print(f"加载原始数据: {len(df)} 条记录, {len(df.columns)} 个字段")
    print(f"字段列表: {list(df.columns)}")
    return df

def preprocess_to_12channels(df):
    """
    将AI4I原始数据转换为12通道监测数据格式
    对应实验五的 motor1 ~ motor12 结构
    """
    print("\n=== 转换为12通道格式 ===")
    
    # 原始核心传感器数据
    air_temp = df['Air temperature [K]'].values
    process_temp = df['Process temperature [K]'].values
    rpm = df['Rotational speed [rpm]'].values
    torque = df['Torque [Nm]'].values
    tool_wear = df['Tool wear [min]'].values
    
    # 构造12个监测通道（从原始特征派生，模拟多传感器采集）
    channels = {}
    
    # motor1: 空气温度（标准化到振动幅值范围）
    channels['motor1'] = (air_temp - np.mean(air_temp)) / np.std(air_temp)
    
    # motor2: 过程温度
    channels['motor2'] = (process_temp - np.mean(process_temp)) / np.std(process_temp)
    
    # motor3: 转速
    channels['motor3'] = (rpm - np.mean(rpm)) / np.std(rpm)
    
    # motor4: 扭矩
    channels['motor4'] = (torque - np.mean(torque)) / np.std(torque)
    
    # motor5: 工具磨损
    channels['motor5'] = (tool_wear - np.mean(tool_wear)) / np.std(tool_wear)

    # 构建DataFrame
    result_df = pd.DataFrame(channels)
    
    # 添加时间戳（模拟按时间顺序采集）
    start_time = datetime(2024, 6, 15, 8, 0, 0)
    timestamps = [start_time + timedelta(minutes=i) for i in range(len(df))]
    result_df.insert(0, 'timestamp', timestamps)
    
    # 故障标签映射
    # label1: 故障大类
    # label2: 故障子类/严重程度
    result_df['label1'] = 0  # 默认正常
    result_df['label2'] = 0
    
    # 映射各故障类型
    # 0: 正常, 1: TWF工具磨损, 2: HDF散热故障, 3: PWF功率故障, 4: OSF过载故障, 5: RNF随机故障
    fault_mapping = {
        'TWF': (1, 1, '工具磨损故障'),
        'HDF': (2, 1, '散热故障'),
        'PWF': (3, 1, '功率故障'),
        'OSF': (4, 1, '过载故障'),
        'RNF': (5, 1, '随机故障'),
    }
    
    result_df['fault_type'] = 'normal'
    result_df['fault_desc'] = '正常运行'
    
    for fault_col, (l1, l2, desc) in fault_mapping.items():
        mask = df[fault_col] == 1
        result_df.loc[mask, 'label1'] = l1
        result_df.loc[mask, 'label2'] = l2
        result_df.loc[mask, 'fault_type'] = fault_col.lower()
        result_df.loc[mask, 'fault_desc'] = desc
    
    # 处理多重故障（优先标记主要故障）
    multi_fault = df['Machine failure'] == 1
    print(f"总故障样本数: {df['Machine failure'].sum()}")
    print(f"故障类型分布:")
    for fault_col in ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']:
        print(f"  {fault_col}: {df[fault_col].sum()} 个")
    
    # 添加产品类型字段
    result_df['product_type'] = df['Type'].values
    result_df['uid'] = df['UDI'].values
    
    print(f"\n转换完成，共 {len(result_df)} 条记录")
    print(f"正常样本: {(result_df['label1'] == 0).sum()}")
    print(f"故障样本: {(result_df['label1'] != 0).sum()}")
    
    return result_df

def save_processed_data(df):
    """保存处理后的数据"""
    output_path = os.path.join(get_data_dir(), 'raw_motor_data.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n原始数据已保存到: {output_path}")
    return output_path

def main():
    print("=" * 60)
    print("步骤1：数据采集 - AI4I 2020 真实工业数据集")
    print("=" * 60)
    
    # 1. 下载数据集
    filepath = download_dataset()
    
    if filepath is None:
        print("无法下载数据集，请手动下载 ai4i2020.csv 放到 data/ 目录下")
        print("下载地址: https://archive.ics.uci.edu/dataset/601/ai4i")
        return None
    
    # 2. 加载原始数据
    raw_df = load_raw_data(filepath)
    
    # 3. 预处理为12通道格式
    processed_df = preprocess_to_12channels(raw_df)
    
    # 4. 保存
    save_processed_data(processed_df)
    
    print("\n数据预览:")
    print(processed_df[['timestamp', 'motor1', 'motor2', 'motor3', 'fault_desc']].head(10))
    
    return processed_df

if __name__ == '__main__':
    main()
