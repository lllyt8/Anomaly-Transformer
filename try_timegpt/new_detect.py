import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os
from tqdm import tqdm
from nixtla import NixtlaClient

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# mkdir the folder for saving the testing result
results_dir = "nixtla_results"  # Named the dir
os.makedirs(results_dir, exist_ok=True)  # If the dir already existed, we can use it.

# Get TimeGPT API Key
nixtla_client = NixtlaClient(api_key='nixak-qaTEyzVzAXINDnW5dIkHG3bzsYfdmnRWlcNaav8wRfp9F9xhzbtP3i8hltsRRVMh5MQYOKgSmPHXH3q3')

# Load data
print("Loading data...")
df = pd.read_csv('data_factory/raw_data/JNM.csv', low_memory=False)

# Transformed all the column's name to lower case
df.columns = df.columns.str.lower()
print("Transformed all the column's name to lower case.")

# 设置需要分析的特征列表
features_to_analyze = [
    'systemvolt', 'totalcurrenta', 'soc', 'soh',
    'hcellv', 'lcellv', 'averagecellv', 'cellvdelta',
    'htempc', 'ltempc', 'averagecelltempc', 'tempcdelta'
]

# 确保时间戳列格式正确，使用宽容模式
df['ts'] = pd.to_datetime(df['ts'], errors='coerce')

# 检查缺失的时间戳
missing_ts = df['ts'].isna().sum()
if missing_ts > 0:
    print(f"警告: {missing_ts} 行的时间戳无法解析")
    # 丢弃这些行
    df = df.dropna(subset=['ts'])

# 确认这些特征都在数据集中
available_features = [f for f in features_to_analyze if f in df.columns]
if len(available_features) < len(features_to_analyze):
    missing_features = set(features_to_analyze) - set(available_features)
    print(f"警告: 以下特征在数据集中不存在: {missing_features}")
    features_to_analyze = available_features

# 获取所有电池串ID
battery_ids = df['stringid'].unique()
print(f"共发现 {len(battery_ids)} 个电池串ID")

# 创建汇总报告的数据结构
summary_results = {
    'battery_id': [],
    'feature': [],
    'total_points': [],
    'anomalies_detected': [],
    'anomaly_percentage': []
}

# 对每个电池串和每个特征进行异常检测
for battery_id in tqdm(battery_ids, desc="处理电池串"):
    for feature in features_to_analyze:
        try:
            # 只选择特定电池串的数据
            battery_df = df[df['stringid'] == battery_id].copy()
            
            # 检查是否有足够的数据点
            if len(battery_df) < 10:
                print(f"  警告: 电池串 {battery_id} 只有 {len(battery_df)} 个数据点，跳过")
                continue
            
            # 创建一个只包含时间戳和当前特征的新数据框
            detection_df = pd.DataFrame()
            detection_df['ds'] = battery_df['ts']
            detection_df['y'] = battery_df[feature]
            
            # 去除缺失值
            detection_df = detection_df.dropna()
            
            # 检查是否还有足够的数据点
            if len(detection_df) < 10:
                print(f"  警告: 电池串 {battery_id} 的 {feature} 特征有太多缺失值，跳过")
                continue
            
            # 检查时间戳是否有重复
            dup_times = detection_df.duplicated(subset=['ds'], keep=False)
            if dup_times.any():
                print(f"  信息: 电池串 {battery_id} 的 {feature} 特征有 {dup_times.sum()} 个重复时间戳，取均值")
                # 取均值
                detection_df = detection_df.groupby('ds')['y'].mean().reset_index()
            
            # 按时间戳排序
            detection_df = detection_df.sort_values('ds')
            
            # 重新采样以确保时间间隔一致
            # 首先将ds设为索引
            detection_df = detection_df.set_index('ds')
            
            # 确定适当的频率
            time_diff = pd.Series(detection_df.index).diff().median()
            if time_diff.total_seconds() < 60:
                freq = '1min'
            elif time_diff.total_seconds() < 3600:
                freq = '1h'
            else:
                freq = '1D'
            
            # 重新采样
            detection_df = detection_df.resample(freq).mean()
            
            # 填充缺失值
            detection_df = detection_df.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
            
            # 重置索引
            detection_df = detection_df.reset_index()
            
            # 确保没有NaN值
            if detection_df['y'].isna().any():
                print(f"  警告: 电池串 {battery_id} 的 {feature} 特征有无法填充的NaN值，跳过")
                continue
            
            # 检测异常
            try:
                # 调试输出
                if battery_id == battery_ids[0] and feature == features_to_analyze[0]:
                    print("检测数据样例:")
                    print(detection_df.head())
                    detection_df.head().to_csv(f"{results_dir}/detection_sample_{battery_id}_{feature}.csv", index=False)
                
                # 调用TimeGPT的异常检测功能
                try:
                    # 尝试指定freq参数
                    anomalies_df = nixtla_client.detect_anomalies(
                        df=detection_df,
                        freq=freq
                    )
                except Exception as freq_error:
                    print(f"  使用freq参数失败: {str(freq_error)}，尝试不使用freq参数")
                    # 如果失败，尝试不指定freq参数
                    anomalies_df = nixtla_client.detect_anomalies(
                        df=detection_df
                    )
                
                # 保存异常检测结果
                if anomalies_df is not None and len(anomalies_df) > 0:
                    anomalies_df.to_csv(f"{results_dir}/anomalies_{battery_id}_{feature}.csv", index=False)
                    
                    # 计算异常比例
                    anomaly_count = anomalies_df['anomaly'].sum()
                    anomaly_percentage = (anomaly_count / len(anomalies_df)) * 100
                    
                    # 更新汇总结果
                    summary_results['battery_id'].append(battery_id)
                    summary_results['feature'].append(feature)
                    summary_results['total_points'].append(len(anomalies_df))
                    summary_results['anomalies_detected'].append(anomaly_count)
                    summary_results['anomaly_percentage'].append(anomaly_percentage)
                    
                    print(f"  成功: 电池串 {battery_id} 的 {feature} 特征检测到 {anomaly_count} 个异常点，占比 {anomaly_percentage:.2f}%")
                    
                    # 绘制结果（只为第一个电池串的每个特征绘制）
                    if battery_id == battery_ids[0]:
                        try:
                            plt.figure(figsize=(10, 6))
                            nixtla_client.plot(
                                df=detection_df,
                                fcst_df=anomalies_df,
                                plot_anomalies=True
                            )
                            plt.title(f'电池串 {battery_id} - {feature} 异常检测结果')
                            plt.tight_layout()
                            plt.savefig(f"{results_dir}/plot_{battery_id}_{feature}.png")
                            plt.close()
                        except Exception as plot_error:
                            print(f"  无法绘制 {battery_id} 的 {feature} 特征图表: {str(plot_error)}")
                    
                else:
                    print(f"  警告: 电池串 {battery_id} 的 {feature} 特征异常检测没有返回结果")
                    
            except Exception as api_error:
                print(f"  电池串 {battery_id} 的 {feature} 特征API调用失败: {str(api_error)}")
            
        except Exception as e:
            print(f"  处理电池串 {battery_id} 的 {feature} 特征时出错: {str(e)}")

# 创建并保存汇总报告
summary_df = pd.DataFrame(summary_results)

# 检查汇总数据框是否有数据
print("\n汇总数据框统计信息:")
print(f"- 行数: {len(summary_df)}")
if len(summary_df) == 0:
    print("警告: 没有成功的异常检测结果，无法生成汇总报告和可视化")
else:
    # 打印更多统计信息
    print(f"- 电池串数量: {summary_df['battery_id'].nunique()}")
    print(f"- 特征数量: {summary_df['feature'].nunique()}")
    print(f"- 平均异常百分比: {summary_df['anomaly_percentage'].mean():.2f}%")
    print(f"- 异常百分比范围: {summary_df['anomaly_percentage'].min():.2f}% 到 {summary_df['anomaly_percentage'].max():.2f}%")
    
    # 保存汇总报告
    summary_df.to_csv(f"{results_dir}/anomaly_detection_summary.csv", index=False)
    
    # 生成汇总可视化
    print("\n生成汇总可视化...")
    
    # 按特征绘制异常百分比条形图
    try:
        feature_summary = summary_df.groupby('feature')['anomaly_percentage'].mean().reset_index()
        
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='feature', y='anomaly_percentage', data=feature_summary, palette='viridis')
        plt.title('各特征平均异常百分比 (%)', fontsize=14)
        plt.xlabel('特征', fontsize=12)
        plt.ylabel('异常百分比 (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # 添加数值标签
        for i, v in enumerate(feature_summary['anomaly_percentage']):
            ax.text(i, v + 0.5, f"{v:.2f}%", ha='center', fontsize=10)
            
        plt.tight_layout()
        plt.savefig(f"{results_dir}/feature_anomaly_percentage.png")
        plt.close()
    except Exception as e:
        print(f"生成特征条形图时出错: {str(e)}")
    
    # 按电池串ID绘制异常百分比条形图
    try:
        battery_summary = summary_df.groupby('battery_id')['anomaly_percentage'].mean().reset_index()
        
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='battery_id', y='anomaly_percentage', data=battery_summary, palette='rocket')
        plt.title('各电池串平均异常百分比 (%)', fontsize=14)
        plt.xlabel('电池串ID', fontsize=12)
        plt.ylabel('异常百分比 (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # 添加数值标签
        for i, v in enumerate(battery_summary['anomaly_percentage']):
            ax.text(i, v + 0.5, f"{v:.2f}%", ha='center', fontsize=10)
            
        plt.tight_layout()
        plt.savefig(f"{results_dir}/battery_anomaly_percentage.png")
        plt.close()
    except Exception as e:
        print(f"生成电池串条形图时出错: {str(e)}")
    
    # 按特征和电池串ID绘制异常比例热图
    try:
        pivot_df = summary_df.pivot_table(
            values='anomaly_percentage', 
            index='battery_id', 
            columns='feature',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0)
        plt.title('各电池串各特征的异常百分比 (%)')
        plt.tight_layout()
        plt.savefig(f"{results_dir}/anomaly_percentage_heatmap.png")
        plt.close()
    except Exception as e:
        print(f"生成热图时出错: {str(e)}")
