"""
工业互联网课程大作业 - 步骤5：大屏可视化数据导出脚本
1. 修复滑动窗口看不到故障问题，取消时间正序重排
2. 增加故障标签校验，label1≠0强制故障描述，避免数值异常显示正常
3. 适配前端dashboard.html，修复fault_types无fault_desc字段报错
"""
import sqlite3
import pandas as pd
import json
import os

def get_db_path():
    return os.path.join(os.path.dirname(__file__), '..', 'database', 'industrial_iot.db')

def export_dashboard_data():
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    dashboard_data = {}

    # 1. 趋势曲线 最新200条，取消sort，最新数据在数组头部
    trend = pd.read_sql_query('''
        SELECT timestamp, motor1, motor2, motor3, motor4, motor5, fault_desc, label1
        FROM motor_oscillation_data 
        ORDER BY timestamp DESC LIMIT 200
    ''', conn)

    if len(trend) > 0:
        # 新增：标签校验，只要label1≠0，强制标记故障，防止数值异常但描述正常
        def fix_fault_desc(row):
            if row["label1"] != 0 and row["fault_desc"] == "正常运行":
                return "设备异常故障"
            return row["fault_desc"]
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
        dashboard_data['trend'] = {
            'timestamps': [], 'motor1': [], 'motor2': [], 'motor3': [], 'motor4': [], 'motor5': [],
            'fault_states': [], 'speed_group': [], 'wear_group': [], 'stress_group': []
        }

    # 2. 传感器通道中文名映射
    dashboard_data['channel_names'] = {
        'motor1': '空气温度',
        'motor2': '过程温度',
        'motor3': '转速',
        'motor4': '扭矩',
        'motor5': '工具磨损'
    }

    # 3. 数据概览卡片
    overview = pd.read_sql_query('''
        SELECT 
            COUNT(*) as total_records,
            SUM(CASE WHEN label1 = 0 THEN 1 ELSE 0 END) as normal_count,
            SUM(CASE WHEN label1 != 0 THEN 1 ELSE 0 END) as failure_count
        FROM motor_oscillation_data
    ''', conn)
    if len(overview) > 0:
        dashboard_data['overview'] = overview.iloc[0].to_dict()
    else:
        dashboard_data['overview'] = {
            'total_records': 0,
            'normal_count': 0,
            'failure_count': 0
        }

    # 4. 故障统计饼图数据
    fault_stats = pd.read_sql_query('''
        SELECT fault_desc, label1, LOWER(fault_type) as fault_type,
               COUNT(*) as count
        FROM motor_oscillation_data 
        GROUP BY fault_desc, label1, fault_type
        ORDER BY count DESC
    ''', conn)
    dashboard_data['fault_stats'] = fault_stats.to_dict('records')

    # 5. RMS特征对比柱状图
    rms_data = pd.read_sql_query('''
        SELECT fault_desc,
               SQRT(AVG(motor1*motor1)) as rms_temp,
               SQRT(AVG(motor3*motor3)) as rms_speed,
               SQRT(AVG(motor4*motor4)) as rms_torque,
               SQRT(AVG(motor5*motor5)) as rms_wear,
               SQRT(AVG(motor2*motor2)) as rms_power
        FROM motor_oscillation_data 
        GROUP BY fault_desc
        ORDER BY fault_desc
    ''', conn)
    dashboard_data['rms_comparison'] = rms_data.to_dict('records') if len(rms_data) > 0 else []

    # 6. 故障类型表【修复：移除不存在的fault_desc】
    fault_types = pd.read_sql_query('''
        SELECT fault_code, fault_name, fault_severity, label1
        FROM fault_types 
        ORDER BY label1
    ''', conn)
    if len(fault_types) > 0:
        fault_types['fault_code'] = fault_types['fault_code'].str.lower()
    dashboard_data['fault_types'] = fault_types.to_dict('records')

    # 7. 模型指标
    model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'model_results.json')
    if os.path.exists(model_path):
        with open(model_path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
            if 'feature_importance' in model_data:
                for feat in model_data['feature_importance']:
                    if 'feature' not in feat and 'name' in feat:
                        feat['feature'] = feat['name']
            dashboard_data['model_results'] = model_data
    else:
        dashboard_data['model_results'] = {
            "random_forest": {"accuracy": 0},
            "fault_types": 0,
            "feature_importance": []
        }

    conn.close()

    # 导出json
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'visualization')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'dashboard_data.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dashboard_data, f, ensure_ascii=False, indent=2)

    print(f"导出完成：{output_path}")
    print("数据模块：", list(dashboard_data.keys()))
    print(f"趋势数据条数：{len(dashboard_data['trend']['timestamps'])}")
    return dashboard_data

if __name__ == '__main__':
    export_dashboard_data()