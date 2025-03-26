import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
from tqdm import tqdm
import matplotlib.dates as mdates

# Allow showing Chinese in Graph correctly by using "Heiti TC" font style
plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# Data path
original_data_file = 'data_factory/raw_data/JNM.csv'

# Result dir
results_dir = "nixtla_results"

# New dir, store result on analyzing detection result
anomaly_analysis_dir = os.path.join(results_dir, "anomaly_analysis")
os.makedirs(anomaly_analysis_dir, exist_ok=True)

# Load OG data
print("Loading OG data...")
df_original = pd.read_csv(original_data_file, low_memory=False)
df_original.columns = df_original.columns.str.lower()
df_original['ts'] = pd.to_datetime(df_original['ts'], errors='coerce')

# Get all detection result csv files
anomaly_files = glob.glob(os.path.join(results_dir, "anomalies_*.csv"))
print(f"Found {len(anomaly_files)} detection result csv files")

# 创建一个空的DataFrame来存储所有异常点
all_anomalies = pd.DataFrame()

# 遍历所有异常检测结果文件
for anomaly_file in tqdm(anomaly_files, desc="处理异常文件"):
    try:
        # 从文件名解析电池ID和特征名
        filename = os.path.basename(anomaly_file)
        parts = filename.replace("anomalies_", "").replace(".csv", "").split("_")
        
        if len(parts) != 2:
            print(f"警告: 无法解析文件名 {filename}")
            continue
        
        battery_id, feature = parts
        
        # 读取异常检测结果
        anomalies_df = pd.read_csv(anomaly_file)
        
        # 确保时间戳格式正确
        anomalies_df['ds'] = pd.to_datetime(anomalies_df['ds'])
        
        # 筛选出异常点
        anomaly_points = anomalies_df[anomalies_df['anomaly'] == True].copy()
        
        # 如果没有异常点，跳过
        if len(anomaly_points) == 0:
            continue
        
        # 添加电池ID和特征信息
        anomaly_points['battery_id'] = battery_id
        anomaly_points['feature'] = feature
        
        # 将异常点添加到总的异常点DataFrame中
        all_anomalies = pd.concat([all_anomalies, anomaly_points])
        
        # 特定电池ID和特征的数据
        battery_data = df_original[df_original['stringid'] == int(battery_id)].copy()
        
        # 确保特征列在battery_data中存在
        if feature not in battery_data.columns:
            print(f"警告: 特征 {feature} 在原始数据中不存在，跳过")
            continue
        
        # 排序数据 - 这是解决网状图问题的关键
        battery_data = battery_data.sort_values('ts')
        anomalies_df = anomalies_df.sort_values('ds')
        
        # 为每个特征创建一个异常点可视化
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 使用散点图绘制原始数据而不是线图
        ax.scatter(battery_data['ts'], battery_data[feature], 
                color='blue', s=5, alpha=0.6, label='原始数据')
        
        # 绘制TimGPT预测值
        ax.plot(anomalies_df['ds'], anomalies_df['TimeGPT'], 
            'g-', linewidth=2, alpha=0.7, label='预测值')
        
        # 异常点
        ax.scatter(anomaly_points['ds'], anomaly_points['y'], 
                color='red', s=50, label='异常点')
        
        # 设置日期格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        plt.title(f'电池 {battery_id} - {feature} 异常点')
        plt.xlabel('时间')
        plt.ylabel('值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(os.path.join(anomaly_analysis_dir, f"anomaly_plot_{battery_id}_{feature}.png"))
        plt.close()
        
    except Exception as e:
        print(f"处理文件 {anomaly_file} 时出错: {e}")

# 保存所有异常点
if len(all_anomalies) > 0:
    all_anomalies.to_csv(os.path.join(anomaly_analysis_dir, "all_anomalies.csv"), index=False)
    print(f"已保存 {len(all_anomalies)} 个异常点到 {os.path.join(anomaly_analysis_dir, 'all_anomalies.csv')}")
    
    # 创建一个电池ID和特征的交叉表，显示各自的异常点数量
    anomaly_counts = all_anomalies.groupby(['battery_id', 'feature']).size().unstack().fillna(0).astype(int)
    anomaly_counts.to_csv(os.path.join(anomaly_analysis_dir, "anomaly_counts.csv"))
    print("已生成异常点计数交叉表")
    
    # 创建热图显示异常点分布
    plt.figure(figsize=(14, 10))
    plt.title("各电池各特征异常点数量")
    sns.heatmap(anomaly_counts, annot=True, fmt="d", cmap="YlGnBu")
    plt.tight_layout()
    plt.savefig(os.path.join(anomaly_analysis_dir, "anomaly_counts_heatmap.png"))
    plt.close()
    
    # 额外统计分析 - 按电池串统计异常点总数
    battery_summary = all_anomalies.groupby('battery_id').size().reset_index(name='异常点数量')
    battery_summary.to_csv(os.path.join(anomaly_analysis_dir, "battery_anomaly_summary.csv"), index=False)
    
    # 额外统计分析 - 按特征统计异常点总数
    feature_summary = all_anomalies.groupby('feature').size().reset_index(name='异常点数量')
    feature_summary.to_csv(os.path.join(anomaly_analysis_dir, "feature_anomaly_summary.csv"), index=False)
    
    # 创建条形图
    plt.figure(figsize=(12, 6))
    sns.barplot(x='battery_id', y='异常点数量', data=battery_summary)
    plt.title('各电池串异常点总数')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(anomaly_analysis_dir, "battery_anomaly_count.png"))
    plt.close()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='feature', y='异常点数量', data=feature_summary)
    plt.title('各特征异常点总数')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(anomaly_analysis_dir, "feature_anomaly_count.png"))
    plt.close()
else:
    print("警告: 没有找到任何异常点")

print("异常点分析完成")
