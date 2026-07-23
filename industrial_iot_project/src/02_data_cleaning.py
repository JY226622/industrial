"""
工业互联网课程大作业 - 步骤2：原始数据集采集与数据清洗脚本
适配数据集：UCI AI4I 2020 工业预测性维护标准数据集
数据源官方地址：https://archive.ics.uci.edu/dataset/601/ai4i
整体功能说明：
1. 自动从UCI服务器下载原始数据集csv文件，本地不存在则自动下载，存在直接复用
2. 读取原始传感器物理指标，标准化生成5路核心监测通道motor1~motor5
3. 生成连续时序时间戳，模拟工业设备每分钟采样一条数据
4. 根据原始故障标记TWF/HDF/PWF/OSF/RNF映射故障分类标签、故障名称、故障描述
5. 拼接业务字段（产品型号、设备唯一编号），区分正常/故障样本
6. 输出清洗完成的标准化csv文件，供后续数据库入库脚本读取使用
约束说明：仅保留5个原生物理传感器通道，不额外生成6~12派生振动通道
"""
# 数据处理库
import pandas as pd
import numpy as np
# 时间生成工具，构造时序采样时间戳
from datetime import datetime, timedelta
# 路径管理、文件夹创建
import os
# 网络请求，在线下载UCI数据集
import urllib.request

# UCI官方AI4I2020数据集直链
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/ai4i2020.csv"


def get_data_dir():
    """
    获取项目data文件夹绝对路径
    脚本存放路径：项目根目录/scr/data_clean.py
    数据存放路径：项目根目录/data/
    :return: str data文件夹完整路径
    """
    return os.path.join(os.path.dirname(__file__), '..', 'data')


def download_dataset():
    """
    在线下载AI4I 2020原始数据集
    逻辑：先判断本地是否存在csv文件，存在直接返回路径；不存在则发起网络下载
    异常捕获：网络下载失败返回None，提示用户手动放置数据集
    :return: str 原始数据集文件完整路径 / None 下载失败
    """
    data_dir = get_data_dir()
    # 若data文件夹不存在，自动创建文件夹
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    # 原始数据集本地存储完整路径
    filepath = os.path.join(data_dir, 'ai4i2020_raw.csv')

    # 文件已存在，无需重复下载
    if os.path.exists(filepath):
        print(f"本地已存在原始数据集，跳过下载: {filepath}")
        return filepath

    print(f"开始从UCI机器学习库下载AI4I 2020工业数据集...")
    print(f"下载源地址: {DATASET_URL}")

    try:
        # 网络请求下载文件并写入本地
        urllib.request.urlretrieve(DATASET_URL, filepath)
        print(f"数据集下载完成，保存路径: {filepath}")
        return filepath
    except Exception as e:
        # 捕获网络异常、链接失效等问题
        print(f"数据集在线下载失败，错误信息: {e}")
        print("提示：请手动下载ai4i2020.csv放入项目data文件夹")
        return None


def load_raw_data(filepath):
    """
    读取本地原始数据集csv文件
    :param filepath: 原始csv文件路径
    :return: DataFrame 未清洗原始全量数据集
    """
    df = pd.read_csv(filepath)
    # 打印原始数据集基础信息
    print(f"成功加载原始数据集：共 {len(df)} 条采样记录，{len(df.columns)} 个原始字段")
    print(f"原始数据集字段列表: {list(df.columns)}")
    return df


def preprocess_to_5channels(df):
    """
    核心数据清洗与标准化转换函数
    功能：
    1. 提取5类物理传感器原始数值，Z-score标准化生成motor1~motor5通道
    2. 生成每分钟间隔的连续时间戳，模拟工业实时采集时序
    3. 初始化故障标签，根据原始故障标记匹配故障编号、类型、中文描述
    4. 拼接业务标识字段：产品型号product_type、设备唯一编号uid
    通道映射规则：
    motor1 = 空气温度 Air temperature [K]
    motor2 = 过程温度 Process temperature [K]
    motor3 = 主轴转速 Rotational speed [rpm]
    motor4 = 加工扭矩 Torque [Nm]
    motor5 = 工具磨损时长 Tool wear [min]
    :param df: 原始数据集DataFrame
    :return: DataFrame 标准化5通道清洗完成数据集
    """
    print("\n=== 开始标准化转换，生成5路传感器监测通道 ===")

    # 1. 提取原始5个物理传感器指标
    air_temp = df['Air temperature [K]'].values
    process_temp = df['Process temperature [K]'].values
    rpm = df['Rotational speed [rpm]'].values
    torque = df['Torque [Nm]'].values
    tool_wear = df['Tool wear [min]'].values

    channels = {}
    # Z-score标准化：(原始值-均值)/标准差，消除量纲差异，统一数值分布
    channels['motor1'] = (air_temp - np.mean(air_temp)) / np.std(air_temp)
    channels['motor2'] = (process_temp - np.mean(process_temp)) / np.std(process_temp)
    channels['motor3'] = (rpm - np.mean(rpm)) / np.std(rpm)
    channels['motor4'] = (torque - np.mean(torque)) / np.std(torque)
    channels['motor5'] = (tool_wear - np.mean(tool_wear)) / np.std(tool_wear)

    # 构建5通道基础数据表
    result_df = pd.DataFrame(channels)

    # 2. 生成时序时间戳：起始时间2024-06-15 08:00，每条数据间隔1分钟
    start_time = datetime(2024, 6, 15, 8, 0, 0)
    timestamps = [start_time + timedelta(minutes=i) for i in range(len(df))]
    # 将时间戳插入表格第一列
    result_df.insert(0, 'timestamp', timestamps)

    # 3. 初始化故障相关字段，默认全部为正常运行状态
    result_df['label1'] = 0        # 故障大类编码，0=正常，1~5对应五类故障
    result_df['label2'] = 0        # 辅助故障标记位
    result_df['fault_type'] = 'normal'  # 故障英文短标识
    result_df['fault_desc'] = '正常运行' # 故障中文描述

    # 原始故障编码与标签、文本映射字典
    # key：原始数据集故障列名；value：(label1编码, label2标记, 故障中文名称)
    fault_mapping = {
        'TWF': (1, 1, '工具磨损故障'),
        'HDF': (2, 1, '散热故障'),
        'PWF': (3, 1, '功率故障'),
        'OSF': (4, 1, '过载故障'),
        'RNF': (5, 1, '随机故障'),
    }

    # 遍历五类故障，批量更新对应样本的故障标签信息
    for fault_col, (l1, l2, desc) in fault_mapping.items():
        # 筛选当前故障的全部样本行
        mask = df[fault_col] == 1
        result_df.loc[mask, 'label1'] = l1
        result_df.loc[mask, 'label2'] = l2
        result_df.loc[mask, 'fault_type'] = fault_col.lower()
        result_df.loc[mask, 'fault_desc'] = desc

    # 统计并打印各类故障样本数量
    print(f"数据集总故障样本数量: {df['Machine failure'].sum()}")
    print(f"各类故障样本分布统计:")
    for fault_col in ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']:
        print(f"  {fault_col}: {df[fault_col].sum()} 条")

    # 4. 附加业务字段：产品型号、设备唯一ID
    result_df['product_type'] = df['Type'].values
    result_df['uid'] = df['UDI'].values

    # 输出清洗后数据集概况
    print(f"\n数据标准化转换完成，总记录条数：{len(result_df)}")
    print(f"正常运行样本数量: {(result_df['label1'] == 0).sum()}")
    print(f"故障设备样本数量: {(result_df['label1'] != 0).sum()}")
    print(f"输出仅保留5个核心监测通道：motor1, motor2, motor3, motor4, motor5")

    return result_df


def save_processed_data(df):
    """
    将清洗、标准化完成的5通道数据集保存为csv文件
    :param df: 清洗后的完整数据集
    :return: str 输出文件完整路径
    """
    output_path = os.path.join(get_data_dir(), 'clean_motor_data.csv')
    # index=False 不输出pandas自带行索引
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n清洗完成的5通道数据集已保存至: {output_path}")
    return output_path


def main():
    """
    脚本主执行函数，完整数据清洗流水线执行顺序：
    1. 下载/读取原始AI4I数据集
    2. 加载原始数据并打印基础信息
    3. 标准化、时序构造、故障标签映射，生成5通道清洗数据
    4. 保存清洗后csv文件
    5. 打印前10行数据预览，方便校验清洗效果
    """
    print("=" * 60)
    print("步骤2：工业数据集采集与清洗流程启动 - AI4I 2020")
    print("=" * 60)

    # 步骤1：获取原始数据集文件路径（自动下载/本地读取）
    filepath = download_dataset()

    # 下载失败终止流程，提示手动放置数据集
    if filepath is None:
        print("数据集加载失败，无法继续清洗流程")
        print("解决方案：手动下载ai4i2020.csv放入项目data文件夹")
        print("官方下载地址: https://archive.ics.uci.edu/dataset/601/ai4i")
        return None

    # 步骤2：读取原始csv数据
    raw_df = load_raw_data(filepath)

    # 步骤3：核心清洗标准化处理
    processed_df = preprocess_to_5channels(raw_df)

    # 步骤4：导出清洗后文件，供数据库入库脚本使用
    save_processed_data(processed_df)

    # 打印前10行数据预览，校验通道、时间、故障字段是否正常
    print("\n清洗后数据前10行预览（时间+5通道+故障描述）:")
    print(processed_df[['timestamp', 'motor1', 'motor2', 'motor3', 'motor4', 'motor5', 'fault_desc']].head(10))

    return processed_df


# 脚本直接运行时，自动执行完整数据清洗流程
if __name__ == '__main__':
    main()