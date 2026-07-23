"""
工业互联网大作业 - 主运行脚本
一键执行所有步骤：数据生成 -> 数据清洗 -> 数据库存储 -> 数据分析 -> 导出大屏数据
"""
import os
import sys

# 添加src目录到路径
sys.path.insert(0, os.path.dirname(__file__))

def main():
    print("=" * 60)
    print("工业互联网大作业 - 电机故障诊断与监控系统")
    print("=" * 60)
    
    # 步骤1：数据生成
    print("\n【步骤1/5】数据生成...")
    from src import 01_data_generation
    _01_data_generation.generate_dataset()
    
    # 步骤2：数据清洗
    print("\n【步骤2/5】数据清洗与预处理...")
    from src import 02_data_cleaning
    _02_data_cleaning.main()
    
    # 步骤3：数据库存储
    print("\n【步骤3/5】数据库存储...")
    from src import 03_database_storage
    _03_database_storage.main()
    
    # 步骤4：数据分析与机器学习
    print("\n【步骤4/5】数据分析与故障诊断模型...")
    from src import 04_data_analysis
    _04_data_analysis.main()
    
    # 步骤5：导出大屏数据
    print("\n【步骤5/5】导出可视化大屏数据...")
    from src import _05_export_dashboard_data
    _05_export_dashboard_data.export_dashboard_data()
    
    print("\n" + "=" * 60)
    print("所有步骤执行完成！")
    print("=" * 60)
    print("\n项目文件结构:")
    print("  data/          - 原始数据、清洗数据、特征数据")
    print("  database/      - SQLite数据库文件")
    print("  src/           - Python源代码（5个步骤）")
    print("  visualization/ - ECharts可视化大屏")
    print("  report/        - 实验报告")
    print("\n可视化大屏打开方式:")
    print("  直接用浏览器打开 visualization/dashboard.html")
    print("\n运行各步骤:")
    print("  python src/01_data_generation.py")
    print("  python src/02_data_cleaning.py")
    print("  python src/03_database_storage.py")
    print("  python src/04_data_analysis.py")
    print("  python src/05_export_dashboard_data.py")

if __name__ == '__main__':
    main()
