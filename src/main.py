"""
工业互联网课程大作业 - 全流程精简整合版
功能：数据清洗 → 数据库存储 → 数据分析建模 → 大屏数据导出
"""
import pandas as pd
import numpy as np
import sqlite3
import os
import json
import urllib.request
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 【全局常量配置】所有路径、映射、通道名唯一入口
# ============================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DB_PATH = os.path.join(BASE_DIR, 'database', 'industrial_iot.db')
VIS_DIR = os.path.join(BASE_DIR, 'visualization')
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
START_TIME = datetime(2026, 7, 15, 8, 0, 0)

# 5个核心监测通道
MOTOR_COLS = ['motor1', 'motor2', 'motor3', 'motor4', 'motor5']
CHANNEL_NAMES = {
    'motor1': '空气温度', 'motor2': '过程温度',
    'motor3': '转速',     'motor4': '扭矩',
    'motor5': '工具磨损'
}

# 故障全局映射
FAULT_MAPPING = {
    'TWF': (1, 1, '工具磨损故障'),
    'HDF': (2, 1, '散热故障'),
    'PWF': (3, 1, '功率故障'),
    'OSF': (4, 1, '过载故障'),
    'RNF': (5, 1, '随机故障'),
}
FAULT_LABEL_NAME = {
    0: '正常', 1: '工具磨损', 2: '散热故障',
    3: '功率故障', 4: '过载故障', 5: '随机故障'
}

# 故障字典表数据
FAULT_DICT = [
    (0, 0, 'NORMAL', '正常运行', '无', '设备正常运行状态'),
    (1, 1, 'TWF', '工具磨损故障', '轻微', '刀具磨损达到阈值，需要更换'),
    (2, 1, 'HDF', '散热故障', '严重', '温差过小导致散热失效，设备过热'),
    (3, 1, 'PWF', '功率故障', '中等', '功率异常，转速与扭矩不匹配'),
    (4, 1, 'OSF', '过载故障', '严重', '过载应力超过设备极限'),
    (5, 1, 'RNF', '随机故障', '轻微', '偶发随机故障'),
]


# ============================================================
# 【工具函数层】统一封装
# ============================================================
def ensure_dir(path):
    """确保文件夹存在"""
    if not os.path.exists(path):
        os.makedirs(path)


def download_dataset():
    """下载AI4I 2020数据集，已存在则跳过"""
    ensure_dir(DATA_DIR)
    filepath = os.path.join(DATA_DIR, 'ai4i2020_raw.csv')
    if os.path.exists(filepath):
        print(f"数据集已存在: {filepath}")
        return filepath
    print(f"正在从UCI下载AI4I 2020数据集...")
    try:
        urllib.request.urlretrieve(DATASET_URL, filepath)
        print(f"下载完成: {filepath}")
        return filepath
    except Exception as e:
        print(f"下载失败: {e}")
        print("请手动下载ai4i2020.csv放入data文件夹")
        return None


def write_json(data, save_path):
    """统一JSON写入，中文不转码"""
    ensure_dir(os.path.dirname(save_path))
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(file_path):
    """统一JSON读取"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================
# 【数据库封装层】统一SQLite操作
# ============================================================
class IndustrialDB:
    def __init__(self):
        ensure_dir(os.path.dirname(DB_PATH))
        self.conn = sqlite3.connect(DB_PATH)
        self.cursor = self.conn.cursor()

    def query_df(self, sql):
        """查询返回DataFrame"""
        return pd.read_sql_query(sql, self.conn)

    def execute(self, sql, params=None):
        """执行增删改"""
        if params:
            self.cursor.execute(sql, params)
        else:
            self.cursor.execute(sql)
        self.conn.commit()

    def batch_insert(self, sql, data_list):
        """批量插入"""
        self.cursor.executemany(sql, data_list)
        self.conn.commit()

    def close(self):
        self.conn.close()


# ============================================================
# 【模块1：数据清洗】
# ============================================================
def clean_data():
    """数据清洗主流程：下载 → 读取 → 标准化 → 故障标签映射 → 保存"""
    print("=" * 60)
    print("步骤1：工业数据集采集与清洗 - AI4I 2020")
    print("=" * 60)

    filepath = download_dataset()
    if filepath is None:
        return None

    # 读取原始数据
    df = pd.read_csv(filepath)
    print(f"加载原始数据: {len(df)} 条记录, {len(df.columns)} 个字段")

    # 1. 提取5个物理传感器指标，Z-score标准化
    print("\n=== 标准化转换，生成5路监测通道 ===")
    raw_cols = [
        'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
    ]
    channels = {}
    for motor, col in zip(MOTOR_COLS, raw_cols):
        vals = df[col].values
        channels[motor] = (vals - np.mean(vals)) / np.std(vals)

    result_df = pd.DataFrame(channels)

    # 2. 生成时序时间戳（每分钟一条）
    timestamps = [START_TIME + timedelta(minutes=i) for i in range(len(df))]
    result_df.insert(0, 'timestamp', timestamps)

    # 3. 故障标签映射
    result_df['label1'] = 0
    result_df['label2'] = 0
    result_df['fault_type'] = 'normal'
    result_df['fault_desc'] = '正常运行'

    for fault_col, (l1, l2, desc) in FAULT_MAPPING.items():
        mask = df[fault_col] == 1
        result_df.loc[mask, 'label1'] = l1
        result_df.loc[mask, 'label2'] = l2
        result_df.loc[mask, 'fault_type'] = fault_col.lower()
        result_df.loc[mask, 'fault_desc'] = desc

    # 4. 业务字段
    result_df['product_type'] = df['Type'].values
    result_df['uid'] = df['UDI'].values

    # 5. 保存清洗后数据
    output_path = os.path.join(DATA_DIR, 'clean_motor_data.csv')
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"总故障样本数: {df['Machine failure'].sum()}")
    print(f"正常样本: {(result_df['label1'] == 0).sum()}")
    print(f"故障样本: {(result_df['label1'] != 0).sum()}")
    print(f"清洗完成，已保存: {output_path}")
    print("\n数据预览:")
    print(result_df[['timestamp', 'motor1', 'motor2', 'motor3', 'fault_desc']].head(5))
    return result_df


# ============================================================
# 【模块2：数据库存储】
# ============================================================
def init_database():
    """创建数据库表结构 + 初始化故障字典"""
    db = IndustrialDB()

    # 清理旧表
    for table in ['motor_oscillation_data', 'fault_types', 'motor_stat_features',
                  'raw_sensor_data', 'analysis_results']:
        db.execute(f'DROP TABLE IF EXISTS {table}')

    # 1. 电机振动数据表（5通道）
    db.execute('''
    CREATE TABLE motor_oscillation_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        motor1 REAL NOT NULL, motor2 REAL NOT NULL,
        motor3 REAL NOT NULL, motor4 REAL NOT NULL, motor5 REAL NOT NULL,
        label1 INTEGER NOT NULL, label2 INTEGER NOT NULL,
        fault_type TEXT, fault_desc TEXT, product TEXT, uid INTEGER
    )
    ''')

    # 2. 故障类型字典表
    db.execute('''
    CREATE TABLE fault_types (
        label1 INTEGER, label2 INTEGER,
        fault_code TEXT PRIMARY KEY,
        fault_name TEXT, fault_severity TEXT, description TEXT
    )
    ''')
    db.batch_insert('INSERT INTO fault_types VALUES (?, ?, ?, ?, ?, ?)', FAULT_DICT)

    # 3. 统计特征表
    db.execute('''
    CREATE TABLE motor_stat_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fault_type TEXT, fault_desc TEXT, label1 INTEGER, label2 INTEGER,
        sample_count INTEGER, feature_json TEXT
    )
    ''')

    # 4. 原始传感器数据表
    db.execute('''
    CREATE TABLE raw_sensor_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        uid INTEGER, product_type TEXT,
        air_temperature REAL, process_temperature REAL,
        rotational_speed REAL, torque REAL, tool_wear REAL,
        machine_failure INTEGER, timestamp DATETIME
    )
    ''')

    # 5. 分析结果表
    db.execute('''
    CREATE TABLE analysis_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_type TEXT, result_json TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    db.close()
    print(f"数据库创建成功: {DB_PATH}")


def import_to_db():
    """导入清洗数据 + 原始数据到数据库"""
    print("\n" + "=" * 60)
    print("步骤2：数据入库 - SQLite")
    print("=" * 60)

    init_database()
    db = IndustrialDB()

    # 1. 导入清洗后的5通道数据
    clean_path = os.path.join(DATA_DIR, 'clean_motor_data.csv')
    df = pd.read_csv(clean_path, parse_dates=['timestamp'])
    db.execute('DELETE FROM motor_oscillation_data')

    insert_data = []
    for _, row in df.iterrows():
        insert_data.append((
            row['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            float(row['motor1']), float(row['motor2']),
            float(row['motor3']), float(row['motor4']), float(row['motor5']),
            int(row['label1']), int(row['label2']),
            row['fault_type'], row['fault_desc'],
            row.get('product_type', ''), int(row.get('uid', 0))
        ))

    db.batch_insert('''
    INSERT INTO motor_oscillation_data
    (timestamp, motor1, motor2, motor3, motor4, motor5,
     label1, label2, fault_type, fault_desc, product, uid)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', insert_data)

    count = db.cursor.execute('SELECT COUNT(*) FROM motor_oscillation_data').fetchone()[0]
    print(f"成功导入 {count} 条5通道传感器数据")

    # 2. 导入原始AI4I数据
    raw_path = os.path.join(DATA_DIR, 'ai4i2020_raw.csv')
    if os.path.exists(raw_path):
        raw_df = pd.read_csv(raw_path)
        db.execute('DELETE FROM raw_sensor_data')
        raw_insert = []
        for idx, row in raw_df.iterrows():
            ts = pd.Timestamp(START_TIME) + pd.Timedelta(minutes=idx)
            raw_insert.append((
                int(row['UDI']), row['Type'],
                float(row['Air temperature [K]']),
                float(row['Process temperature [K]']),
                float(row['Rotational speed [rpm]']),
                float(row['Torque [Nm]']),
                float(row['Tool wear [min]']),
                int(row['Machine failure']),
                ts.strftime('%Y-%m-%d %H:%M:%S')
            ))
        db.batch_insert('''
        INSERT INTO raw_sensor_data VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', raw_insert)
        print(f"导入原始传感器数据: {len(raw_insert)} 条")

    # 3. 故障分布统计
    fault_stats = db.cursor.execute('''
        SELECT fault_desc, COUNT(*) FROM motor_oscillation_data
        GROUP BY fault_desc ORDER BY COUNT(*) DESC
    ''').fetchall()
    print("\n各故障类型数据量:")
    for desc, cnt in fault_stats:
        print(f"  {desc}: {cnt} 条")

    db.close()


# ============================================================
# 【模块3：数据分析建模】
# ============================================================
def run_analysis():
    """数据分析全流程：统计 → 相关性 → 二分类 → 多分类 → 结果保存"""
    print("\n" + "=" * 60)
    print("步骤3：数据分析与机器学习故障诊断")
    print("=" * 60)

    db = IndustrialDB()
    df = db.query_df("SELECT * FROM motor_oscillation_data")
    db.close()
    print(f"加载数据: {len(df)} 条记录")

    # ---- 1. 描述性统计 ----
    print("\n=== 统计分析 ===")
    print("\n整体统计描述:")
    print(df[MOTOR_COLS].describe().round(4))

    print("\n各故障类型样本分布:")
    print(df['fault_desc'].value_counts())

    print("\n各故障振动强度(RMS):")
    for fault in df['fault_desc'].unique():
        fd = df[df['fault_desc'] == fault]
        rms = [np.sqrt(np.mean(fd[m] ** 2)) for m in MOTOR_COLS[:3]]
        print(f"  {fault}: m1={rms[0]:.3f}, m2={rms[1]:.3f}, m3={rms[2]:.3f}")

    # ---- 2. 相关性分析 ----
    print("\n=== 相关性分析 ===")
    corr_matrix = df[MOTOR_COLS].corr()
    print("5通道相关系数矩阵:")
    print(corr_matrix.round(3))

    corr_pairs = []
    for i in range(len(MOTOR_COLS)):
        for j in range(i + 1, len(MOTOR_COLS)):
            corr_pairs.append((MOTOR_COLS[i], MOTOR_COLS[j], corr_matrix.iloc[i, j]))
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    print(f"\n相关性最高的3组: {[(p[0], p[1], round(p[2], 4)) for p in corr_pairs[:3]]}")
    print(f"相关性最低的3组: {[(p[0], p[1], round(p[2], 4)) for p in corr_pairs[-3:]]}")

    # ---- 3. 二分类故障检测 ----
    print("\n=== 二分类故障检测（正常/故障） ===")
    y_binary = (df['label1'] != 0).astype(int)
    X = df[MOTOR_COLS].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.3, random_state=42, stratify=y_binary
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    rf_bin = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_bin.fit(X_train_s, y_train)
    bin_acc = accuracy_score(y_test, rf_bin.predict(X_test_s))
    print(f"二分类准确率: {bin_acc:.4f}")
    print(classification_report(y_test, rf_bin.predict(X_test_s),
                                target_names=['正常', '故障'], zero_division=0))

    # ---- 4. 多分类故障诊断 ----
    print("\n=== 多分类故障诊断模型训练 ===")
    y = df['label1'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler2 = StandardScaler()
    X_train_s2 = scaler2.fit_transform(X_train)
    X_test_s2 = scaler2.transform(X_test)

    target_names = [FAULT_LABEL_NAME[i] for i in sorted(FAULT_LABEL_NAME.keys())]

    # 随机森林
    print("\n--- 随机森林 ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train_s2, y_train)
    rf_pred = rf.predict(X_test_s2)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"准确率: {rf_acc:.4f}")
    print(classification_report(y_test, rf_pred, target_names=target_names, zero_division=0))

    feat_imp = pd.DataFrame({
        'feature': MOTOR_COLS, 'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\n特征重要性:")
    for _, row in feat_imp.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # SVM
    print("\n--- SVM ---")
    svm = SVC(kernel='rbf', random_state=42, class_weight='balanced')
    svm.fit(X_train_s2, y_train)
    svm_pred = svm.predict(X_test_s2)
    svm_acc = accuracy_score(y_test, svm_pred)
    print(f"SVM准确率: {svm_acc:.4f}")

    cm = confusion_matrix(y_test, rf_pred)
    print("\n随机森林混淆矩阵:")
    print(cm)

    # ---- 5. 结果持久化 ----
    results = {
        'dataset': 'AI4I 2020 Predictive Maintenance Dataset',
        'total_samples': len(df),
        'fault_types': len(FAULT_LABEL_NAME),
        'random_forest': {
            'accuracy': float(rf_acc),
            'classification_report': classification_report(
                y_test, rf_pred, output_dict=True, zero_division=0)
        },
        'svm': {'accuracy': float(svm_acc)},
        'feature_importance': feat_imp.to_dict('records'),
        'confusion_matrix': cm.tolist(),
        'fault_names': FAULT_LABEL_NAME
    }

    result_path = os.path.join(DATA_DIR, 'model_results.json')
    write_json(results, result_path)
    print(f"\n模型结果已保存: {result_path}")

    # 写入数据库
    db = IndustrialDB()
    db.execute('''
    INSERT INTO analysis_results (analysis_type, result_json)
    VALUES (?, ?)
    ''', ('fault_diagnosis', json.dumps(results, ensure_ascii=False)))
    db.close()

    # ---- 6. 综合摘要 ----
    print("\n=== 数据分析综合摘要 ===")
    print(f"总样本: {len(df)} | 故障类型: {df['fault_type'].nunique()}类")
    print(f"正常: {(df['label1']==0).sum()} | 故障: {(df['label1']!=0).sum()}")
    print(f"最优模型: {'随机森林' if rf_acc >= svm_acc else 'SVM'}")
    print(f"最高准确率: {max(rf_acc, svm_acc):.2%}")
    print(f"TOP3特征: {feat_imp['feature'].head(3).tolist()}")

    return results


# ============================================================
# 【模块4：大屏数据导出】
# ============================================================
def export_dashboard():
    """导出大屏可视化所需的全部JSON数据"""
    print("\n" + "=" * 60)
    print("步骤4：大屏可视化数据导出")
    print("=" * 60)

    db = IndustrialDB()
    dashboard_data = {}

    # 1. 趋势曲线（最新200条）
    trend = db.query_df('''
        SELECT timestamp, motor1, motor2, motor3, motor4, motor5, fault_desc, label1
        FROM motor_oscillation_data ORDER BY timestamp DESC LIMIT 200
    ''')

    def fix_fault_desc(row):
        if row["label1"] != 0 and row["fault_desc"] == "正常运行":
            return "设备异常故障"
        return row["fault_desc"]

    if len(trend) > 0:
        trend["fault_desc"] = trend.apply(fix_fault_desc, axis=1)
        timestamps = pd.to_datetime(trend['timestamp']).dt.strftime('%m-%d %H:%M').tolist()
        dashboard_data['trend'] = {
            'timestamps': timestamps,
            'motor1': trend['motor1'].round(4).tolist(),
            'motor2': trend['motor2'].round(4).tolist(),
            'motor3': trend['motor3'].round(4).tolist(),
            'motor4': trend['motor4'].round(4).tolist(),
            'motor5': trend['motor5'].round(4).tolist(),
            'fault_states': trend['fault_desc'].tolist(),
            'speed_group': trend['motor3'].round(4).tolist(),
            'wear_group': trend['motor5'].round(4).tolist(),
            'stress_group': trend['motor4'].round(4).tolist()
        }
    else:
        dashboard_data['trend'] = {k: [] for k in
            ['timestamps', 'motor1', 'motor2', 'motor3', 'motor4', 'motor5',
             'fault_states', 'speed_group', 'wear_group', 'stress_group']}

    # 2. 通道中文名
    dashboard_data['channel_names'] = CHANNEL_NAMES

    # 3. 数据概览
    overview = db.query_df('''
        SELECT COUNT(*) as total_records,
               SUM(CASE WHEN label1 = 0 THEN 1 ELSE 0 END) as normal_count,
               SUM(CASE WHEN label1 != 0 THEN 1 ELSE 0 END) as failure_count
        FROM motor_oscillation_data
    ''')
    dashboard_data['overview'] = overview.iloc[0].to_dict() if len(overview) > 0 else \
        {'total_records': 0, 'normal_count': 0, 'failure_count': 0}

    # 4. 故障统计饼图
    fault_stats = db.query_df('''
        SELECT fault_desc, label1, LOWER(fault_type) as fault_type, COUNT(*) as count
        FROM motor_oscillation_data GROUP BY fault_desc, label1, fault_type
        ORDER BY count DESC
    ''')
    dashboard_data['fault_stats'] = fault_stats.to_dict('records')

    # 5. RMS特征对比
    rms_data = db.query_df('''
        SELECT fault_desc,
               SQRT(AVG(motor1*motor1)) as rms_temp,
               SQRT(AVG(motor3*motor3)) as rms_speed,
               SQRT(AVG(motor4*motor4)) as rms_torque,
               SQRT(AVG(motor5*motor5)) as rms_wear,
               SQRT(AVG(motor2*motor2)) as rms_power
        FROM motor_oscillation_data GROUP BY fault_desc ORDER BY fault_desc
    ''')
    dashboard_data['rms_comparison'] = rms_data.to_dict('records') if len(rms_data) > 0 else []

    # 6. 故障类型表
    fault_types = db.query_df('''
        SELECT fault_code, fault_name, fault_severity, label1
        FROM fault_types ORDER BY label1
    ''')
    if len(fault_types) > 0:
        fault_types['fault_code'] = fault_types['fault_code'].str.lower()
    dashboard_data['fault_types'] = fault_types.to_dict('records')

    # 7. 模型指标
    model_path = os.path.join(DATA_DIR, 'model_results.json')
    if os.path.exists(model_path):
        model_data = read_json(model_path)
        if 'feature_importance' in model_data:
            for feat in model_data['feature_importance']:
                if 'feature' not in feat and 'name' in feat:
                    feat['feature'] = feat['name']
        dashboard_data['model_results'] = model_data
    else:
        dashboard_data['model_results'] = {
            "random_forest": {"accuracy": 0},
            "fault_types": 0, "feature_importance": []
        }

    db.close()

    # 导出JSON
    output_path = os.path.join(VIS_DIR, 'dashboard_data.json')
    write_json(dashboard_data, output_path)
    print(f"导出完成：{output_path}")
    print("数据模块：", list(dashboard_data.keys()))
    print(f"趋势数据条数：{len(dashboard_data['trend']['timestamps'])}")
    return dashboard_data


# ============================================================
# 【主入口】一键执行全流程
# ============================================================
def main():
    """一键执行完整流水线：清洗 → 入库 → 分析 → 大屏导出"""
    # 步骤1：数据清洗
    clean_data()
    # 步骤2：数据库存储
    import_to_db()
    # 步骤3：数据分析建模
    run_analysis()
    # 步骤4：大屏数据导出
    export_dashboard()



if __name__ == '__main__':
    main()