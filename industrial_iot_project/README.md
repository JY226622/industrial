# 工业互联网大作业 - 设备故障诊断与监控系统

## 项目简介
本项目是工业互联网课程大作业，采用**纯本地化开源方案**，使用UCI真实工业数据集，无需任何云平台费用，完整实现工业大数据从采集到可视化的全流程。

## 数据集
**AI4I 2020 Predictive Maintenance Dataset**
- 来源：UCI Machine Learning Repository（标准工业预测性维护数据集）
- 规模：10,000条真实工业场景数据
- 故障类型：工具磨损、散热故障、功率故障、过载故障、随机故障
- 原始指标：空气温度、过程温度、转速、扭矩、工具磨损

## 技术栈
- **数据处理**：Python + Pandas + NumPy + SciPy
- **数据存储**：SQLite
- **机器学习**：Scikit-learn
- **可视化**：ECharts 5.4

## 功能模块
1. **数据采集** - 自动下载UCI AI4I 2020真实数据集
2. **数据转换** - 5个原始指标派生为12通道监测数据（对应实验五motor1~motor12）
3. **数据清洗** - 缺失值处理、异常值检测、移动平均滤波、特征提取
4. **数据存储** - SQLite数据库持久化存储（5张数据表）
5. **数据分析** - 统计分析、相关性分析、随机森林/SVM故障诊断
6. **可视化** - ECharts工业监控大屏

## 快速开始

### 1. 安装依赖
```bash
pip install pandas numpy scipy scikit-learn
```

### 2. 一键运行
```bash
python run_all.py
```

### 3. 查看可视化大屏
直接用浏览器打开 `visualization/dashboard.html`

## 项目结构
```
industrial_iot_project/
├── run_all.py                    # 一键运行脚本
├── README.md                     # 项目说明
├── data/                         # 数据文件
│   ├── ai4i2020_raw.csv          # UCI原始数据集
│   ├── raw_motor_data.csv        # 12通道转换后数据
│   ├── clean_motor_data.csv      # 清洗后数据
│   └── motor_features.csv        # 统计特征
├── database/                     # 数据库
│   └── industrial_iot.db         # SQLite数据库
├── src/                          # 源代码（5个步骤）
│   ├── 01_data_generation.py
│   ├── 02_data_cleaning.py
│   ├── 03_database_storage.py
│   ├── 04_data_analysis.py
│   └── 05_export_dashboard_data.py
├── visualization/                # 可视化大屏
│   ├── dashboard.html            # ECharts大屏
│   └── dashboard_data.json       # 大屏数据
└── report/                       # 实验报告
    └── 实验报告.md
```

## 12通道监测指标
| 通道 | 名称 | 说明 |
|------|------|------|
| motor1 | 空气温度 | 环境温度 |
| motor2 | 过程温度 | 加工温度 |
| motor3 | 转速 | 电机转速 |
| motor4 | 扭矩 | 输出扭矩 |
| motor5 | 工具磨损 | 刀具磨损量 |
| motor6 | 温差 | 过程-空气温度差 |

## 故障类型
1. 正常运行
2. 工具磨损故障 (TWF)
3. 散热故障 (HDF)
4. 功率故障 (PWF)
5. 过载故障 (OSF)
6. 随机故障 (RNF)

## 大屏预览
可视化大屏采用深色科技风格，三栏布局：
- **左侧**：数据概览、12通道实时指标、设备状态
- **中间**：运行参数趋势图、故障特征对比柱状图
- **右侧**：当前状态、故障分布饼图、模型指标、故障字典
