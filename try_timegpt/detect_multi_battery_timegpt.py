import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from nixtla import NixtlaClient

# mkdir the folder for saving the testing result
results_dir = "nixtla_results"  # Named the dir
os.makedirs(results_dir, exist_ok=True)  # If the dir already existed, we can use it.

# Get TimeGPT API Key
nixtla_client = NixtlaClient(api_key='nixak-qaTEyzVzAXINDnW5dIkHG3bzsYfdmnRWlcNaav8wRfp9F9xhzbtP3i8hltsRRVMh5MQYOKgSmPHXH3q3')

# Load data
print("Loading data...")
df = pd.read_csv('data_factory/raw_data/JNM.csv')

# Transformed all the column's name to lower case
df.columns = df.columns.str.lower()
print("Transformed all the column's name to lower case.")

# 同样修改特征列表为小写
features_to_analyze = [
    'systemvolt', 'totalcurrenta', 'soc', 'soh',
    'hcellv', 'lcellv', 'averagecellv', 'cellvdelta',
    'htempc', 'ltempc', 'averagecelltempc', 'tempcdelta'
]


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

# 确认这些特征都在数据集中
available_features = [f for f in features_to_analyze if f in df.columns]
if len(available_features) < len(features_to_analyze):
    missing_features = set(features_to_analyze) - set(available_features)
    print(f"警告: 以下特征在数据集中不存在: {missing_features}")
    features_to_analyze = available_features

# 获取所有电池串ID
battery_ids = df['stringId'].unique()
print(f"共发现 {len(battery_ids)} 个电池串ID")

# 分析每个电池串的时间间隔
print("分析数据采样频率...")
all_freqs = {}
for battery_id in battery_ids:
    battery_data = df[df['stringId'] == battery_id].sort_values('ts')
    
    # 检查是否有足够的数据点
    if len(battery_data) < 2:
        print(f"  警告: 电池串 {battery_id} 只有 {len(battery_data)} 个数据点，跳过频率分析")
        all_freqs[battery_id] = '1h'  # 设置默认频率
        continue
        
    # 计算时间差
    time_diffs = battery_data['ts'].diff().dropna()
    if len(time_diffs) == 0:
        print(f"  警告: 电池串 {battery_id} 无法计算时间差，使用默认频率")
        all_freqs[battery_id] = '1h'  # 设置默认频率
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
    print(f"\n分析特征: {feature}")
    
    # 为每个电池串进行异常检测
    for battery_id in tqdm(battery_ids, desc=f"处理 {feature} 的电池串"):
        try:
            # 只选择特定电池串的数据
            id_df = df[df['stringId'] == battery_id][['ts', feature]].copy()
            
            # 检查是否有足够的数据点
            if len(id_df) < 10:
                print(f"  警告: 电池串 {battery_id} 的 {feature} 特征只有 {len(id_df)} 个数据点，跳过")
                continue
                
            # 检查此ID的数据是否有缺失值
            missing = id_df[feature].isnull().sum()
            if missing > 0:
                print(f"  信息: 电池串 {battery_id} 的 {feature} 特征有 {missing} 个缺失值，尝试插值填充")
                # 尝试插值填充
                id_df[feature] = id_df[feature].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
                
                # 填充后再次检查
                remaining_missing = id_df[feature].isnull().sum()
                if remaining_missing > 0:
                    print(f"  警告: 填充后仍有 {remaining_missing} 个缺失值，跳过此电池串")
                    continue
            
            # 检查时间戳是否有重复
            dup_times = id_df.duplicated(subset=['ts'], keep=False)
            if dup_times.any():
                print(f"  信息: 电池串 {battery_id} 有 {dup_times.sum()} 个重复时间戳，取均值")
                # 取均值
                id_df = id_df.groupby('ts')[feature].mean().reset_index()
            
            # 按时间戳排序
            id_df = id_df.sort_values('ts')
            
            # 确定频率
            freq = all_freqs.get(battery_id, '1h')  # 默认为1小时
            
            # 使用规则的时间索引重新采样数据
            start_time = id_df['ts'].min()
            end_time = id_df['ts'].max()
            
            # 创建规则的时间序列
            regular_ts = pd.date_range(start=start_time, end=end_time, freq=freq)
            
            # 检查生成的时间范围是否合理
            if len(regular_ts) < 10:
                print(f"  警告: 基于频率 {freq} 生成的时间范围只有 {len(regular_ts)} 个点，跳过")
                continue
                
            # 设置时间戳为索引，便于重采样
            id_df = id_df.set_index('ts')
            
            # 创建新的规则时间序列数据框
            regular_df = pd.DataFrame(index=regular_ts)
            
            # 将原始数据合并到规则时间序列中
            regular_df = regular_df.join(id_df, how='left')
            
            # 线性插值填充缺失值
            regular_df[feature] = regular_df[feature].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
            
            # 再次检查是否有任何NaN值
            if regular_df[feature].isnull().any():
                print(f"  警告: 插值后仍有缺失值，跳过此电池串")
                continue
                
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
                
                # 检查是否成功获取了异常结果
                if anomalies_df is None or len(anomalies_df) == 0:
                    print(f"  警告: 电池串 {battery_id} 的API调用没有返回任何异常")
                    continue
                    
                # 保存结果
                anomalies_df.to_csv(f"{results_dir}/anomalies_{battery_id}_{feature}.csv", index=False)
                
                # 绘制结果（只为前3个电池串绘制，以节省时间）
                if battery_id in battery_ids[:3]:
                    try:
                        plot = nixtla_client.plot(
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
                
                print(f"  成功: 电池串 {battery_id} 的 {feature} 特征分析完成，检测到 {len(anomalies_df)} 个异常点，占比 {len(anomalies_df) / len(regular_df) * 100:.2f}%")
                
            except Exception as api_error:
                print(f"  电池串 {battery_id} 的API调用失败: {str(api_error)}")
            
        except Exception as e:
            print(f"  处理电池串 {battery_id} 时出错: {str(e)}")

# 创建并保存汇总报告
summary_df = pd.DataFrame(summary_results)

# 检查汇总数据框是否有数据
print("\n汇总数据框统计信息:")
print(f"- 行数: {len(summary_df)}")
if len(summary_df) == 0:
    print("警告: 没有成功的异常检测结果，无法生成汇总报告和可视化")
else:
    # 打印更多统计信息
    print(f"- 特征数量: {summary_df['feature'].nunique()}")
    print(f"- 电池串数量: {summary_df['battery_id'].nunique()}")
    print(f"- 平均异常百分比: {summary_df['anomaly_percentage'].mean():.2f}%")
    print(f"- 异常百分比范围: {summary_df['anomaly_percentage'].min():.2f}% 到 {summary_df['anomaly_percentage'].max():.2f}%")
    
    # 保存汇总报告
    summary_df.to_csv(f"{results_dir}/anomaly_detection_summary.csv", index=False)
    
    # 生成汇总可视化
    print("\n生成汇总可视化...")
    
    # 添加条形图
    # 按特征绘制异常百分比条形图
    try:
        feature_summary = summary_df.groupby('feature')['anomaly_percentage'].mean().reset_index()
        
        if len(feature_summary) > 0:
            plt.figure(figsize=(12, 6))
            # 使用更明显的颜色
            ax = sns.barplot(x='feature', y='anomaly_percentage', data=feature_summary, palette='deep')
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
            print(f"特征条形图已保存: {results_dir}/feature_anomaly_percentage.png")
        else:
            print("警告: 没有足够的特征数据生成条形图")
    except Exception as e:
        print(f"生成特征条形图时出错: {str(e)}")
    
    # 按电池串ID绘制异常百分比条形图
    try:
        battery_summary = summary_df.groupby('battery_id')['anomaly_percentage'].mean().reset_index()
        
        if len(battery_summary) > 0:
            plt.figure(figsize=(12, 6))
            # 使用更明显的颜色
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
            print(f"电池串条形图已保存: {results_dir}/battery_anomaly_percentage.png")
        else:
            print("警告: 没有足够的电池串数据生成条形图")
    except Exception as e:
        print(f"生成电池串条形图时出错: {str(e)}")
    
    # 按特征和电池串ID绘制异常比例热图
        if summary_df['feature'].nunique() == 0 or summary_df['battery_id'].nunique() == 0:
            print("警告: 没有足够的特征或电池串数据来创建热图")
            # 创建一个空热图并添加文本说明
            plt.figure(figsize=(12, 10))
            plt.text(0.5, 0.5, '没有足够数据生成热图', 
                    horizontalalignment='center', 
                    verticalalignment='center',
                    fontsize=16)
            plt.savefig(f"{results_dir}/anomaly_percentage_heatmap_empty.png")
            plt.close()
            print(f"空热图已保存: {results_dir}/anomaly_percentage_heatmap_empty.png")
        else:
            print("生成热图...")
        # 数据透视表
        pivot_df = summary_df.pivot_table(
            values='anomaly_percentage', 
            index='battery_id', 
            columns='feature',
            aggfunc='mean'
        )
        
        # 检查透视表是否有数据
        print(f"透视表形状: {pivot_df.shape}")
        print(f"透视表非空值数量: {pivot_df.count().sum()}")
        
        if pivot_df.shape[0] > 0 and pivot_df.shape[1] > 0 and pivot_df.count().sum() > 0:
            plt.figure(figsize=(12, 10))
            
            # 设置颜色范围，确保低值也能显示
            vmax = max(1.0, pivot_df.max().max())
            
            # 绘制热图
            sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=vmax)
            plt.title('各电池串各特征的异常百分比 (%)')
            plt.tight_layout()
            plt.savefig(f"{results_dir}/anomaly_percentage_heatmap.png")
            plt.close()
            print(f"热图已保存: {results_dir}/anomaly_percentage_heatmap.png")
        else:
            print("警告: 透视表为空或不包含有效数据，无法绘制热图")
            # 创建一个空热图并添加文本说明
            plt.figure(figsize=(12, 10))
            plt.text(0.5, 0.5, '没有足够数据生成热图', 
                    horizontalalignment='center', 
                    verticalalignment='center',
                    fontsize=16)
            plt.savefig(f"{results_dir}/anomaly_percentage_heatmap_empty.png")
            plt.close()
            print(f"空热图已保存: {results_dir}/anomaly_percentage_heatmap_empty.png")
