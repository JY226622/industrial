"""
工业互联网大作业 - 步骤3：数据存储（SQLite数据库）
替代阿里云RDS，使用轻量级SQLite数据库
5个传感器通道motor1~motor5
"""
import sqlite3
import pandas as pd
import os
import json

def get_db_path():
    return os.path.join(os.path.dirname(__file__), '..', 'database', 'industrial_iot.db')

def create_database():
    """创建数据库和表结构（仅5通道motor1-motor5）"""
    db_path = get_db_path()
    db_dir = os.path.dirname(db_path)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 先删除旧表防止残留字段冲突
    cursor.execute("DROP TABLE IF EXISTS motor_oscillation_data")
    cursor.execute("DROP TABLE IF EXISTS fault_types")
    cursor.execute("DROP TABLE IF EXISTS motor_stat_features")
    cursor.execute("DROP TABLE IF EXISTS raw_sensor_data")

    # 1. 电机振动数据表 仅5通道，无motor6~12
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS motor_oscillation_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        motor1 REAL NOT NULL,
        motor2 REAL NOT NULL,
        motor3 REAL NOT NULL,
        motor4 REAL NOT NULL,
        motor5 REAL NOT NULL,
        label1 INTEGER NOT NULL,
        label2 INTEGER NOT NULL,
        fault_type TEXT,
        fault_desc TEXT,
        product TEXT,
        uid INTEGER
    )
    ''')

    # 2. 故障类型字典表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS fault_types (
        label1 INTEGER,
        label2 INTEGER,
        fault_code TEXT PRIMARY KEY,
        fault_name TEXT,
        fault_severity TEXT,
        description TEXT
    )
    ''')

    # 3. 统计特征表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS motor_stat_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fault_type TEXT,
        fault_desc TEXT,
        label1 INTEGER,
        label2 INTEGER,
        sample_count INTEGER,
        feature_json TEXT
    )
    ''')

    # 4. 原始AI4I传感器数据表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS raw_sensor_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        uid INTEGER,
        product_type TEXT,
        air_temperature REAL,
        process_temperature REAL,
        rotational_speed REAL,
        torque REAL,
        tool_wear REAL,
        machine_failure INTEGER,
        timestamp DATETIME
    )
    ''')

    # 故障字典数据
    fault_dict = [
        (0, 0, 'NORMAL', '正常运行', '无', '设备正常运行状态'),
        (1, 1, 'TWF', '工具磨损故障', '轻微', '刀具磨损达到阈值，需要更换'),
        (2, 1, 'HDF', '散热故障', '严重', '温差过小导致散热失效，设备过热'),
        (3, 1, 'PWF', '功率故障', '中等', '功率异常，转速与扭矩不匹配'),
        (4, 1, 'OSF', '过载故障', '严重', '过载应力超过设备极限'),
        (5, 1, 'RNF', '随机故障', '轻微', '偶发随机故障'),
    ]

    cursor.executemany('''
    INSERT OR REPLACE INTO fault_types VALUES (?, ?, ?, ?, ?, ?)
    ''', fault_dict)

    conn.commit()
    conn.close()
    print(f"数据库创建成功: {db_path}")
    print("已创建表: motor_oscillation_data, fault_types, motor_stat_features, raw_sensor_data")

def import_motor_data():
    """导入清洗电机数据，只读取motor1~motor5，丢弃6-12通道"""
    db_path = get_db_path()
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clean_motor_data')
    df = pd.read_csv(data_path + ".csv", parse_dates=['timestamp'])

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    conn.execute('DELETE FROM motor_oscillation_data')

    # 只取前5个电机通道
    motor_cols = ["motor1", "motor2", "motor3", "motor4", "motor5"]
    insert_data = []

    for _, row in df.iterrows():
        insert_data.append((
            row['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            float(row['motor1']),
            float(row['motor2']),
            float(row['motor3']),
            float(row['motor4']),
            float(row['motor5']),
            int(row['label1']),
            int(row['label2']),
            row['fault_type'],
            row['fault_desc'],
            row.get('product_type', ''),
            int(row.get('uid', 0))
        ))

    # SQL字段与上面元组严格一一对应，无多余motor6字段
    sql = '''
    INSERT INTO motor_oscillation_data 
    (timestamp, motor1, motor2, motor3, motor4, motor5, label1, label2, fault_type, fault_desc, product, uid)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''
    cursor.executemany(sql, insert_data)
    conn.commit()

    count = cursor.execute('SELECT COUNT(*) FROM motor_oscillation_data').fetchone()[0]
    print(f"成功导入 {count} 条5通道传感器数据")

    fault_stats = cursor.execute('''
        SELECT fault_desc, COUNT(*) as cnt 
        FROM motor_oscillation_data 
        GROUP BY fault_desc
        ORDER BY cnt DESC
    ''').fetchall()

    print("\n各故障类型数据量:")
    for desc, cnt in fault_stats:
        print(f"  {desc}: {cnt} 条")

    conn.close()

def import_raw_sensor_data():
    """导入原始AI4I数据集"""
    db_path = get_db_path()
    raw_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'ai4i2020_raw.csv')
    if not os.path.exists(raw_path):
        print("原始数据文件不存在，跳过原始数据表导入")
        return

    df = pd.read_csv(raw_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM raw_sensor_data')

    insert_data = []
    start_time = pd.Timestamp('2024-06-15 08:00:00')
    for idx, row in df.iterrows():
        ts = start_time + pd.Timedelta(minutes=idx)
        insert_data.append((
            int(row['UDI']),
            row['Type'],
            float(row['Air temperature [K]']),
            float(row['Process temperature [K]']),
            float(row['Rotational speed [rpm]']),
            float(row['Torque [Nm]']),
            float(row['Tool wear [min]']),
            int(row['Machine failure']),
            ts.strftime('%Y-%m-%d %H:%M:%S')
        ))

    cursor.executemany('''
    INSERT INTO raw_sensor_data 
    (uid, product_type, air_temperature, process_temperature, rotational_speed, torque, tool_wear, machine_failure, timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', insert_data)
    conn.commit()
    print(f"导入原始传感器数据: {len(insert_data)} 条")
    conn.close()

def import_features():
    """导入特征统计表，修复参数缺失BUG"""
    db_path = get_db_path()
    feature_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'motor_features.csv')
    df = pd.read_csv(feature_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM motor_stat_features')

    for _, row in df.iterrows():
        feature_dict = row.drop(['fault_type', 'fault_desc', 'label1', 'label2', 'sample_count']).to_dict()
        cursor.execute('''
        INSERT INTO motor_stat_features (fault_type, fault_desc, label1, label2, sample_count, feature_json)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            row['fault_type'],
            row['fault_desc'],
            int(row['label1']),
            int(row['label2']),
            int(row['sample_count']),
            json.dumps(feature_dict)
        ))

    conn.commit()
    count = cursor.execute('SELECT COUNT(*) FROM motor_stat_features').fetchone()[0]
    print(f"成功导入 {count} 条特征数据")
    conn.close()

def query_demo():
    """查询演示，修复SQL括号语法错误"""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("\n=== 数据库查询演示 ===")

    print("\n1. 最新10条监测数据:")
    rows = cursor.execute('''
        SELECT timestamp, motor1, motor2, motor3, fault_desc 
        FROM motor_oscillation_data 
        ORDER BY timestamp DESC LIMIT 10
    ''').fetchall()
    for row in rows[:3]:
        print(f"  {row[0]} | m1={row[1]:.4f} m2={row[2]:.4f} m3={row[3]} | {row[4]}")
    print("  ...")

    print("\n2. 各故障通道均值统计:")
    rows = cursor.execute('''
        SELECT fault_desc, COUNT(*), AVG(ABS(motor1)), AVG(motor5)
        FROM motor_oscillation_data 
        GROUP BY fault_desc
        ORDER BY COUNT(*) DESC
    ''').fetchall()
    for row in rows:
        print(f"  {row[0]}: 总量={row[1]}, m1均值={row[2]:.4f}, m5均值={row[3]:.4f}")

    print("\n3. 故障类型字典:")
    rows = cursor.execute('SELECT fault_name, fault_severity, description FROM fault_types').fetchall()
    for row in rows:
        print(f"  {row[0]} - {row[1]} - {row[2]}")

    conn.close()

def main():
    print("=== 步骤3：数据库存储 ===")
    create_database()
    import_motor_data()
    import_raw_sensor_data()
    import_features()
    query_demo()
    print(f"\n数据库存储完成！数据库文件: {get_db_path()}")

if __name__ == '__main__':
    main()