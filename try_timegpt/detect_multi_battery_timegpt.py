import pandas as pd
import numpy as np
from nixtla import NixtlaClient
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from datetime import datetime

# 创建结果目录
results_dir = "nixtla_results"
os.makedirs(results_dir, exist_ok=True)

# 获取API密钥
nixtla_client = NixtlaClient(api_key='nixak-qaTEyzVzAXINDnW5dIkHG3bzsYfdmnRWlcNaav8wRfp9F9xhzbtP3i8hltsRRVMh5MQYOKgSmPHXH3q3')

# 读取数据
print("读取数据...")
df = pd.read_csv('data_factory/raw/DanDLN_Data.csv')

# 分析时间戳格式
print("样本时间戳:", df['ts'].iloc[0])

# 确保时间戳列格式正确 - 使用明确的格式
df['ts'] = pd.to_datetime(df['ts'], format='%m/%d/%y %H:%M', errors='coerce')

# 检查解析后的时间戳
print("解析后的时间戳样本:", df['ts'].iloc[0])

# 检查缺失的时间戳
missing_ts = df['ts'].isna().sum()
if missing_ts > 0:
    print(f"警告: {missing_ts} 行的时间戳无法解析")
    # 丢弃这些行
    df = df.dropna(subset=['ts'])

# 选择要分析的特征
features_to_analyze = [
    'systemVolt', 'totalCurrentA', 'soc', 'soh',
    'hCellV', 'lCellV', 'averageCellV', 'CellVDelta',
    'hTempC', 'lTempC', 'averageCellTempC', 'TempCDelta'
]

# 获取所有电池串ID
battery_ids = df['stringId'].unique()
print(f"共发现 {len(battery_ids)} 个电池串ID")

# 分析每个电池串的时间间隔
print("分析数据采样频率...")
all_freqs = {}
for battery_id in battery_ids:
    battery_data = df[df['stringId'] == battery_id].sort_values('ts')
    
    # 计算时间差
    time_diffs = battery_data['ts'].diff().dropna()
    if len(time_diffs) == 0:
        continue
        
    # 计算最常见的间隔（秒）
    diff_seconds = time_diffs.dt.total_seconds()
    most_common_seconds = diff_seconds.value_counts().idxmax()
    
    # 确定合适的频率字符串
    if most_common_seconds < 60:
        freq = f"{int(most_common_seconds)}s"
    elif most_common_seconds < 3600:
        freq = f"{int(most_common_seconds/60)}min"
    else:
        freq = f"{int(most_common_seconds/3600)}h"
    
    all_freqs[battery_id] = freq
    
# 显示每个电池串的频率
for battery_id, freq in all_freqs.items():
    print(f"电池串 {battery_id} 的频率: {freq}")

# 创建汇总报告的数据结构
summary_results = {
    'battery_id': [],
    'feature': [],
    'total_points': [],
    'anomalies_detected': [],
    'anomaly_percentage': []
}

# 对每个特征和每个电池串进行异常检测
for feature in features_to_analyze:
    if feature not in df.columns:
        print(f"跳过 {feature} - 数据中不存在此列")
        continue
        
    print(f"\n分析特征: {feature}")
    
    # 为每个电池串进行异常检测
    for battery_id in tqdm(battery_ids, desc=f"处理 {feature} 的电池串"):
        try:
            # 只选择特定电池串的数据
            id_df = df[df['stringId'] == battery_id][['ts', feature]].copy()
            
            # 检查此ID的数据是否有缺失值
            missing = id_df[feature].isnull().sum()
            if missing > 0:
                # 尝试插值填充
                id_df[feature] = id_df[feature].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
            
            # 检查时间戳是否有重复
            dup_times = id_df.duplicated(subset=['ts'], keep=False)
            if dup_times.any():
                # 取均值
                id_df = id_df.groupby('ts')[feature].mean().reset_index()
            
            # 按时间戳排序
            id_df = id_df.sort_values('ts')
            
            # 确定频率
            freq = all_freqs.get(battery_id, '1min')  # 默认为1分钟
            
            # 使用规则的时间索引重新采样数据
            start_time = id_df['ts'].min()
            end_time = id_df['ts'].max()
            
            # 创建规则的时间序列
            regular_ts = pd.date_range(start=start_time, end=end_time, freq=freq)
            
            # 设置时间戳为索引，便于重采样
            id_df = id_df.set_index('ts')
            
            # 创建新的规则时间序列数据框
            regular_df = pd.DataFrame(index=regular_ts)
            
            # 将原始数据合并到规则时间序列中
            regular_df = regular_df.join(id_df, how='left')
            
            # 线性插值填充缺失值
            regular_df[feature] = regular_df[feature].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
            
            # 重置索引，将时间戳作为列
            regular_df = regular_df.reset_index()
            regular_df = regular_df.rename(columns={'index': 'ts'})
            
            # 重命名列以匹配API要求
            regular_df = regular_df.rename(columns={feature: 'value'})
            
            # 保存预处理后的数据示例
            if battery_id == battery_ids[0]:
                regular_df.head(50).to_csv(f"{results_dir}/preprocessed_{feature}_sample.csv", index=False)
            
            # 检测异常
            try:
                anomalies_df = nixtla_client.detect_anomalies(
                    regular_df, 
                    time_col='ts', 
                    target_col='value',
                    freq=freq  # 显式指定频率
                )
                
                # 保存结果
                anomalies_df.to_csv(f"{results_dir}/anomalies_{battery_id}_{feature}.csv", index=False)
                
                # 绘制结果（只为前3个电池串绘制，以节省时间）
                if battery_id in battery_ids[:3]:
                    try:
                        nixtla_client.plot(
                            regular_df, 
                            anomalies_df,
                            time_col='ts', 
                            target_col='value'
                        )
                        # 保存图表
                        plt.savefig(f"{results_dir}/plot_{battery_id}_{feature}.png")
                        plt.close()
                    except Exception as plot_error:
                        print(f"  无法绘制 {battery_id} 的图表: {str(plot_error)}")
                
                # 添加到汇总结果
                summary_results['battery_id'].append(battery_id)
                summary_results['feature'].append(feature)
                summary_results['total_points'].append(len(regular_df))
                summary_results['anomalies_detected'].append(len(anomalies_df))
                summary_results['anomaly_percentage'].append(len(anomalies_df) / len(regular_df) * 100 if len(regular_df) > 0 else 0)
                
            except Exception as api_error:
                print(f"  电池串 {battery_id} 的API调用失败: {str(api_error)}")
            
        except Exception as e:
            print(f"  处理电池串 {battery_id} 时出错: {str(e)}")

# 创建并保存汇总报告
summary_df = pd.DataFrame(summary_results)
summary_df.to_csv(f"{results_dir}/anomaly_detection_summary.csv", index=False)

# 生成汇总可视化
print("\n生成汇总可视化...")

# 按特征和电池串ID绘制异常比例热图
if len(summary_df) > 0:
    plt.figure(figsize=(12, 10))
    
    # 数据透视表
    pivot_df = summary_df.pivot_table(
        values='anomaly_percentage', 
        index='battery_id', 
        columns='feature',
        aggfunc='mean'
    )
    
    # 绘制热图
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlOrRd")
    plt.title('各电池串各特征的异常百分比 (%)')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/anomaly_percentage_heatmap.png")
    plt.close()
    
    # 按特征绘制异常百分比条形图
    plt.figure(figsize=(12, 6))
    feature_summary = summary_df.groupby('feature')['anomaly_percentage'].mean().reset_index()
    sns.barplot(x='feature', y='anomaly_percentage', data=feature_summary)
    plt.title('各特征平均异常百分比 (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/feature_anomaly_percentage.png")
    plt.close()
    
    # 按电池串ID绘制异常百分比条形图
    plt.figure(figsize=(12, 6))
    battery_summary = summary_df.groupby('battery_id')['anomaly_percentage'].mean().reset_index()
    sns.barplot(x='battery_id', y='anomaly_percentage', data=battery_summary)
    plt.title('各电池串平均异常百分比 (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/battery_anomaly_percentage.png")
    plt.close()

print(f"\n分析完成！结果保存在 {results_dir} 目录下")
print(f"汇总报告: {results_dir}/anomaly_detection_summary.csv")
print(f"热图: {results_dir}/anomaly_percentage_heatmap.png")
