"""
工业互联网课程大作业 - 步骤4：工业数据分析与机器学习故障诊断
整体功能模块：
1. 数据库读取：从SQLite加载电机振动全量监测数据
2. 描述性统计分析：样本分布、各故障通道RMS振动幅值统计
3. 特征相关性分析：计算5个监测通道相关系数，筛选高低相关特征对
4. 二分类故障检测：区分【正常运行 / 设备故障】二分类任务
5. 多分类故障诊断：随机森林、SVM双模型识别6类故障，输出评估指标
6. 模型结果持久化：导出JSON模型指标文件、写入数据库分析记录表
7. 综合分析摘要输出：汇总数据集、模型精度、关键特征等核心结论
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')


def get_db_path():
    """
    获取工业物联网SQLite数据库绝对路径:return: str 数据库完整文件路径
    """
    return os.path.join(os.path.dirname(__file__), '..', 'database', 'industrial_iot.db')


def load_data_from_db():
    """
    数据库读取函数：读取电机振动监测全量表 motor_oscillation_data
    :return: DataFrame 完整数据集，包含motor1~motor5、故障标签、故障描述等字段
    """
    print("=== 从数据库加载数据 ===")
    db_path = get_db_path()
    # 建立数据库连接
    conn = sqlite3.connect(db_path)
    # 查询整张振动数据表
    df = pd.read_sql_query("SELECT * FROM motor_oscillation_data", conn)
    # 关闭连接释放资源
    conn.close()

    # 打印数据集基础信息
    print(f"加载数据: {len(df)} 条记录")
    print(f"数据集中故障类型总数: {df['fault_type'].nunique()}")
    return df


def statistical_analysis(df):
    """
    描述性统计分析模块
    1. 5个核心传感器通道的均值、方差、最值等统计指标
    2. 各类故障样本数量分布统计
    3. 按故障分组计算前3通道振动RMS均方根幅值（反映振动强度）
    :param df: 全量振动数据集DataFrame
    """
    print("\n=== 统计分析 ===")
    # 监测通道 motor1~motor5
    motor_cols = [f'motor{i}' for i in range(1, 6)]

    # 1. 整体数值分布统计
    print("\n整体统计描述（5个核心监测通道）:")
    stats = df[motor_cols].describe()
    print(stats.round(4))

    # 2. 各故障类型样本计数统计
    print("\n各故障类型样本数量分布:")
    fault_counts = df['fault_desc'].value_counts()
    print(fault_counts)

    # 3. 分组计算每种故障下前3通道RMS振动幅值
    print("\n各故障类型振动强度(RMS值) - motor1/motor2/motor3:")
    for fault in df['fault_desc'].unique():
        # 筛选当前故障所有样本
        fault_data = df[df['fault_desc'] == fault]
        # RMS计算公式：sqrt(特征平方均值)
        rms1 = np.sqrt(np.mean(fault_data['motor1'] ** 2))
        rms2 = np.sqrt(np.mean(fault_data['motor2'] ** 2))
        rms3 = np.sqrt(np.mean(fault_data['motor3'] ** 2))
        print(f"  {fault}: motor1_RMS={rms1:.3f}, motor2_RMS={rms2:.3f}, motor3_RMS={rms3:.3f}")


def correlation_analysis(df):
    """
    特征相关性分析模块
    1. 计算5个传感器通道之间的皮尔逊相关系数矩阵
    2. 筛选相关性绝对值最高、最低的3组通道对，用于特征冗余判断
    :param df: 全量振动数据集DataFrame
    :return: DataFrame 通道相关系数矩阵
    """
    print("\n=== 相关性分析 ===")
    motor_cols = [f'motor{i}' for i in range(1, 6)]

    # 计算多变量相关系数矩阵
    corr_matrix = df[motor_cols].corr()
    print("\n5通道传感器相关系数矩阵:")
    print(corr_matrix.round(3))

    # 遍历所有通道两两组合，存储通道名称与相关系数
    corr_values = []
    for i in range(len(motor_cols)):
        for j in range(i + 1, len(motor_cols)):
            corr_values.append((motor_cols[i], motor_cols[j], corr_matrix.iloc[i, j]))

    # 按相关系数绝对值从大到小排序
    corr_values.sort(key=lambda x: abs(x[2]), reverse=True)
    print(f"\n相关性最高的3组通道组合:")
    for pair in corr_values[:3]:
        print(f"  {pair[0]} - {pair[1]}: {pair[2]:.4f}")

    print(f"\n相关性最低的3组通道组合:")
    for pair in corr_values[-3:]:
        print(f"  {pair[0]} - {pair[1]}: {pair[2]:.4f}")

    return corr_matrix


def build_fault_diagnosis_model(df):
    """
    多分类故障诊断模型训练模块
    任务：基于5个振动通道，区分6类设备状态（正常+5类故障）
    模型：随机森林、SVM支持向量机
    输出：准确率、分类报告、混淆矩阵、特征重要度，并保存完整结果JSON
    :param df: 全量振动数据集DataFrame
    :return: dict 双模型完整评估结果字典
    """
    print("\n=== 故障诊断多分类模型训练 ===")
    motor_cols = [f'motor{i}' for i in range(1, 6)]

    # 1. 划分特征矩阵X、分类标签y(label1代表故障大类编码)
    X = df[motor_cols].values
    y = df['label1'].values

    # 分层划分训练集/测试集，保证训练、测试集中各类故障样本比例一致，避免类别失衡
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"训练集样本量: {len(X_train)}")
    print(f"测试集样本量: {len(X_test)}")
    print(f"训练集各类别样本分布: {dict(pd.Series(y_train).value_counts())}")

    # 2. 特征标准化处理（消除量纲影响，适配SVM、提升模型稳定性）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 故障数字编码与中文名称映射，用于分类报告展示
    fault_names = {
        0: '正常',
        1: '工具磨损',
        2: '散热故障',
        3: '功率故障',
        4: '过载故障',
        5: '随机故障'
    }
    target_names = [fault_names[i] for i in sorted(fault_names.keys())]

    # ---------------------- 模型1：随机森林分类器 ----------------------
    print("\n--- 随机森林分类器训练结果 ---")
    # 100棵决策树，类别平衡权重，固定随机种子保证复现
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)

    # 计算整体预测准确率
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print(f"随机森林总体准确率: {rf_accuracy:.4f}")
    print("\n多分类详细评估报告:")
    print(classification_report(y_test, rf_pred, target_names=target_names, zero_division=0))

    # 提取各传感器通道特征重要性，排序筛选关键特征
    feature_importance = pd.DataFrame({
        'feature': motor_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n特征重要性TOP5:")
    for _, row in feature_importance.head().iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # ---------------------- 模型2：SVM支持向量机分类器 ----------------------
    print("\n--- SVM支持向量机训练结果 ---")
    svm_model = SVC(kernel='rbf', random_state=42, class_weight='balanced')
    svm_model.fit(X_train_scaled, y_train)
    svm_pred = svm_model.predict(X_test_scaled)

    svm_accuracy = accuracy_score(y_test, svm_pred)
    print(f"SVM总体准确率: {svm_accuracy:.4f}")

    # 输出随机森林混淆矩阵，查看各类故障错分情况
    print("\n随机森林混淆矩阵(行=真实标签，列=预测标签):")
    cm = confusion_matrix(y_test, rf_pred)
    print(cm)

    # 组装所有模型评估结果，用于持久化存储
    results = {
        'dataset': 'AI4I 2020 Predictive Maintenance Dataset',
        'total_samples': len(df),
        'fault_types': len(fault_names),
        'random_forest': {
            'accuracy': float(rf_accuracy),
            'classification_report': classification_report(y_test, rf_pred, output_dict=True, zero_division=0)
        },
        'svm': {
            'accuracy': float(svm_accuracy)
        },
        'feature_importance': feature_importance.to_dict('records'),
        'confusion_matrix': cm.tolist(),
        'fault_names': fault_names
    }

    # 将模型结果写入JSON文件，供前端可视化大屏读取
    result_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'model_results.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n模型评估结果JSON已保存至: {result_path}")
    return results


def binary_failure_analysis(df):
    """
    二分类故障检测模块
    简化任务：仅区分【0=正常设备 / 1=任意故障设备】，做基础故障预警
    模型：随机森林
    :param df: 全量振动数据集DataFrame
    :return: float 二分类任务预测准确率
    """
    print("\n=== 二分类故障检测（正常/故障） ===")
    motor_cols = [f'motor{i}' for i in range(1, 6)]

    # 构建二分类标签：label1不等于0则标记为故障(1)，否则正常(0)
    y_binary = (df['label1'] != 0).astype(int)
    X = df[motor_cols].values

    # 分层划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.3, random_state=42, stratify=y_binary
    )

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练随机森林二分类模型
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)

    # 评估输出
    accuracy = accuracy_score(y_test, rf_pred)
    print(f"二分类故障检测整体准确率: {accuracy:.4f}")
    print("二分类详细评估报告:")
    print(classification_report(y_test, rf_pred, target_names=['正常', '故障'], zero_division=0))

    return accuracy


def generate_analysis_report(df, results):
    """
    汇总生成数据分析综合摘要
    整合数据集规模、样本分布、最优模型、关键特征等核心结论
    :param df: 原始数据集DataFrame
    :param results: 多分类模型训练结果字典
    :return: dict 综合摘要数据
    """
    print("\n=== 数据分析综合摘要 ===")
    fault_counts = df['fault_desc'].value_counts()

    # 汇总核心指标
    summary = {
        'dataset': 'AI4I 2020',
        'total_samples': len(df),
        'fault_types_count': df['fault_type'].nunique(),
        'channels': 5,
        'normal_count': int((df['label1'] == 0).sum()),
        'failure_count': int((df['label1'] != 0).sum()),
        # 对比两个模型精度，选出最优模型
        'best_model': '随机森林' if results['random_forest']['accuracy'] >= results['svm']['accuracy'] else 'SVM',
        'best_accuracy': max(results['random_forest']['accuracy'], results['svm']['accuracy']),
        # 提取贡献度最高的3个监测通道
        'top_features': [f['feature'] for f in results['feature_importance'][:3]]
    }

    # 控制台打印摘要信息
    print(f"数据集来源: {summary['dataset']}")
    print(f"总监测样本条数: {summary['total_samples']}")
    print(f"设备故障类别总数: {summary['fault_types_count']}")
    print(f"正常运行样本数量: {summary['normal_count']}")
    print(f"故障样本总数量: {summary['failure_count']}")
    print(f"参与建模监测通道: {summary['channels']}个")
    print(f"故障诊断最优模型: {summary['best_model']}")
    print(f"模型最高预测准确率: {summary['best_accuracy']:.2%}")
    print(f"对故障识别贡献最大的3个通道: {summary['top_features']}")

    return summary


def main():
    """
    程序主入口函数，按顺序执行全流程数据分析任务
    执行流程：加载数据 → 统计分析 → 相关性分析 → 二分类预警 → 多分类故障诊断 → 生成摘要 → 结果入库
    """
    # 步骤1：从SQLite数据库读取振动监测全量数据
    df = load_data_from_db()

    # 步骤2：数据集描述性统计与故障振动幅值分析
    statistical_analysis(df)

    # 步骤3：5个传感器通道相关性计算与特征分析
    correlation_analysis(df)

    # 步骤4：二分类模型（仅区分正常/故障，简易预警）
    binary_failure_analysis(df)

    # 步骤5：多分类故障诊断模型训练，输出评估并保存JSON
    results = build_fault_diagnosis_model(df)

    # 步骤6：生成数据分析综合摘要
    summary = generate_analysis_report(df, results)

    # 将完整模型分析结果存入数据库analysis_results表，持久化留存
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 不存在则创建分析结果存储表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS analysis_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_type TEXT,
        result_json TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # 插入本次故障诊断完整结果JSON字符串
    cursor.execute('''
    INSERT INTO analysis_results (analysis_type, result_json)
    VALUES (?, ?)
    ''', ('fault_diagnosis', json.dumps(results, ensure_ascii=False)))

    # 提交事务并关闭连接
    conn.commit()
    conn.close()

    print("\n=== 全部数据分析流程执行完成 ===")
    print("1. 模型评估指标已导出至 /data/model_results.json")
    print("2. 完整分析结果已存入数据库 analysis_results 表")


# 脚本直接运行时触发主流程
if __name__ == '__main__':
    main()