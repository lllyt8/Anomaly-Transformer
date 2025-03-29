import pandas as pd
import numpy as np
from collections import Counter
import json

# 加载匹配的异常记录
df = pd.read_csv('nixtla_results/anomaly_analysis/complete_anomaly_records_fixed.csv')

# 将字符串列转换为适当的类型
df['ts'] = pd.to_datetime(df['ts'])
df['anomalous_features'] = df['anomalous_features'].fillna('')

# 创建数据摘要
summary = {}

# 1. 基本统计信息
summary['total_anomalies'] = len(df)
summary['unique_data_points'] = df.drop_duplicates(subset=['stringid', 'ts']).shape[0]
summary['time_range'] = [df['ts'].min().strftime('%Y-%m-%d %H:%M'), 
                        df['ts'].max().strftime('%Y-%m-%d %H:%M')]
summary['battery_count'] = df['stringid'].nunique()

# 2. 特征异常分布
feature_counts = {}
for features_str in df['anomalous_features']:
    if features_str:
        for feature in features_str.split(','):
            feature_counts[feature] = feature_counts.get(feature, 0) + 1
summary['feature_counts'] = {k: v for k, v in sorted(feature_counts.items(), 
                                                key=lambda x: x[1], 
                                                reverse=True)}

# 3. 特征共现分析
feature_pairs = {}
for features_str in df['anomalous_features']:
    features = features_str.split(',') if features_str else []
    if len(features) > 1:
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                pair = tuple(sorted([features[i], features[j]]))
                feature_pairs[pair] = feature_pairs.get(pair, 0) + 1
summary['top_feature_pairs'] = {str(k): v for k, v in sorted(feature_pairs.items(), 
                                                        key=lambda x: x[1], 
                                                        reverse=True)[:10]}

# 4. 电池异常对比
battery_anomalies = {}
for battery_id in df['stringid'].unique():
    battery_df = df[df['stringid'] == battery_id]
    features = []
    for features_str in battery_df['anomalous_features']:
        if features_str:
            features.extend(features_str.split(','))
    
    counter = Counter(features)
    battery_anomalies[int(battery_id)] = {
        'total_anomalies': len(battery_df),
        'top_features': dict(counter.most_common(5))
    }
summary['battery_anomalies'] = battery_anomalies

# 5. 时间分布分析
df['hour'] = df['ts'].dt.hour
hourly_anomalies = df.groupby('hour').size().to_dict()
summary['hourly_distribution'] = hourly_anomalies

df['date'] = df['ts'].dt.date
daily_anomalies = df.groupby('date').size().to_dict()
summary['daily_distribution'] = {str(k): v for k, v in daily_anomalies.items()}

# 6. 特征值范围分析
feature_ranges = {}
for feature in feature_counts.keys():
    if feature in df.columns:
        feature_ranges[feature] = {
            'min': float(df[feature].min()),
            'max': float(df[feature].max()),
            'mean': float(df[feature].mean()),
            'median': float(df[feature].median())
        }
summary['feature_ranges'] = feature_ranges

# 7. 分析连续异常
df = df.sort_values(['stringid', 'ts'])
df['prev_ts'] = df.groupby('stringid')['ts'].shift(1)
df['time_diff'] = (df['ts'] - df['prev_ts']).dt.total_seconds() / 60  # 分钟差
consecutive_anomalies = df[df['time_diff'] <= 30].groupby('stringid').size().to_dict()
summary['consecutive_anomalies'] = {int(k): v for k, v in consecutive_anomalies.items()}

# 构建提示词
prompt = """
# 电池异常数据分析

请作为一位锂电池专家，分析以下电池异常数据并提供专业见解。这些数据来自电池管理系统(BMS)的异常检测算法，包含了10个电池组在一周内的异常记录。请根据以下数据摘要，结合锂电池领域的专业知识，分析可能的异常原因、严重程度和建议的措施。

## 数据摘要
## 特征说明
- systemvolt: 系统电压 (V)
- totalcurrenta: 总电流 (A)
- soc: 荷电状态 (%)
- soh: 健康状态 (%)
- hcellv: 最高单体电压 (V)
- lcellv: 最低单体电压 (V) 
- averagecellv: 平均单体电压 (V)
- cellvdelta: 单体电压差 (V)
- htempc: 最高温度 (°C)
- ltempc: 最低温度 (°C)
- averagecelltempc: 平均温度 (°C)
- tempcdelta: 温度差 (°C)

## 分析要点
1. 根据异常特征分布，识别最常见的异常模式
2. 分析特征共现关系，解释可能的物理原因
3. 比较不同电池组的异常情况，找出表现最差的电池组
4. 根据特征值范围分析异常的严重程度
5. 对时间分布模式提供解释（如为什么某些时段异常较多）
6. 提供可能的优化或干预建议

请提供详细的分析报告，包括可能的根本原因和建议的下一步行动。
"""

# 将摘要转换为JSON字符串并插入到提示词中
prompt = prompt.format(summary_json=json.dumps(summary, indent=2))

# 保存提示词到文件，方便输入到LLM
with open('battery_analysis_prompt.txt', 'w') as f:
    f.write(prompt)

print("已生成数据摘要和分析提示词，保存到 battery_analysis_prompt.txt")
