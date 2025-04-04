import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns

def label_battery_anomalies():
    """
    为原始电池数据添加异常标记，基于TimeGPT的异常检测结果，
    并生成工程师验证工作表
    """
    # 定义文件路径
    original_data_file = 'data_factory/raw_data/JNM.csv'
    anomalies_file = 'nixtla_results/anomaly_analysis/all_anomalies.csv'
    output_dir = 'nixtla_results/labeled_data'
    output_file = os.path.join(output_dir, 'labeled_original_data.csv')
    validation_file = os.path.join(output_dir, 'engineer_validation_sheet.csv')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print("加载原始数据...")
    # 加载原始数据
    df_original = pd.read_csv(original_data_file, low_memory=False)
    # 转换列名为小写
    df_original.columns = df_original.columns.str.lower()
    # 处理时间戳
    df_original['ts'] = pd.to_datetime(df_original['ts'], errors='coerce')
    # 移除NaT时间戳的行
    df_original = df_original.dropna(subset=['ts'])
    
    # 检查是否存在异常点文件
    if not os.path.exists(anomalies_file):
        print(f"错误: 无法找到异常点文件 {anomalies_file}")
        return
    
    print("加载异常点数据...")
    try:
        # 加载异常数据
        df_anomalies = pd.read_csv(anomalies_file)
        # 处理时间戳
        df_anomalies['ds'] = pd.to_datetime(df_anomalies['ds'], errors='coerce')
        # 移除NaT时间戳的行
        df_anomalies = df_anomalies.dropna(subset=['ds'])
    except Exception as e:
        print(f"加载异常点数据时出错: {e}")
        return
    
    # 初始化异常标签列，默认为False（非异常）
    df_original['is_anomaly'] = False
    # 初始化异常特征列
    df_original['anomalous_features'] = ''
    # 初始化异常原因列（将用于工程师填写）
    df_original['anomaly_reason'] = ''
    # 初始化人工验证列（将用于工程师填写）
    df_original['manually_verified'] = False
    # 初始化正确性评估列（将用于工程师填写）
    df_original['detection_correct'] = ''
    
    # 创建字典存储异常特征信息
    print("处理异常特征信息...")
    anomaly_features = {}
    for _, row in tqdm(df_anomalies.iterrows(), total=len(df_anomalies), desc="组织异常特征"):
        key = (row['battery_id'], row['ds'])
        if key not in anomaly_features:
            anomaly_features[key] = []
        anomaly_features[key].append(row['feature'])
    
    # 匹配原始数据与异常数据
    print("标记异常点...")
    time_tolerance = timedelta(minutes=1)  # 设置容差为1分钟
    
    # 创建索引以跟踪匹配的点
    matched_indices = set()
    matched_records = []
    
    # 按电池ID分组处理数据
    battery_ids = df_anomalies['battery_id'].unique()
    for battery_id in tqdm(battery_ids, desc="处理电池组"):
        # 提取该电池的原始数据和异常数据
        battery_mask = df_original['stringid'] == battery_id
        if not any(battery_mask):
            print(f"警告: 电池ID {battery_id} 在原始数据中不存在")
            continue
        
        battery_original_indices = df_original[battery_mask].index
        battery_anomalies = df_anomalies[df_anomalies['battery_id'] == battery_id].copy()
        
        # 获取异常点的唯一时间戳
        anomaly_times = battery_anomalies['ds'].unique()
        
        for anomaly_time in anomaly_times:
            # 计算当前电池所有点与异常时间点的时间差
            time_diffs = abs(df_original.loc[battery_original_indices, 'ts'] - anomaly_time)
            
            if not time_diffs.empty:
                min_diff_idx = time_diffs.idxmin()
                min_diff = time_diffs[min_diff_idx]
                
                if min_diff <= time_tolerance:
                    # 获取对应的异常特征
                    key = (battery_id, anomaly_time)
                    if key in anomaly_features:
                        # 在原始数据中标记为异常
                        df_original.loc[min_diff_idx, 'is_anomaly'] = True
                        df_original.loc[min_diff_idx, 'anomalous_features'] = ','.join(anomaly_features[key])
                        matched_indices.add(min_diff_idx)
                        
                        # 收集匹配的记录用于工程师验证
                        matched_record = df_original.loc[min_diff_idx].copy()
                        matched_record['time_diff_seconds'] = min_diff.total_seconds()
                        matched_record['anomaly_time'] = anomaly_time
                        matched_records.append(matched_record)
    
    # 保存标记后的数据
    df_original.to_csv(output_file, index=False)
    
    # 计算统计数据
    anomaly_count = df_original['is_anomaly'].sum()
    total_count = len(df_original)
    anomaly_percentage = (anomaly_count / total_count) * 100
    
    print(f"已保存 {total_count} 条标记后的数据到 {output_file}")
    print(f"异常点总数: {anomaly_count} ({anomaly_percentage:.2f}%)")
    print(f"成功匹配的异常点总数: {len(matched_indices)} (与异常检测结果文件中的 {len(df_anomalies)} 条记录比较)")
    
    # 创建工程师验证工作表
    if matched_records:
        validation_df = pd.DataFrame(matched_records)
        
        # 选择要保留的列
        validation_cols = [
            'stringid', 'ts', 'anomaly_time', 'time_diff_seconds', 'anomalous_features',
            # 以下是工程师需要填写的列
            'manually_verified', 'detection_correct', 'anomaly_reason'
        ]
        
        # 添加所有特征列，以便工程师查看
        feature_cols = [col for col in validation_df.columns if col in features_to_analyze()]
        validation_cols.extend(feature_cols)
        
        # 保留需要的列
        validation_df = validation_df[validation_cols]
        
        # 添加说明列以增强可读性
        validation_df['指导_说明'] = '请在detection_correct列中填写: "正确", "错误" 或 "不确定"'
        
        # 保存验证工作表
        validation_df.to_csv(validation_file, index=False)
        print(f"已为工程师创建验证工作表，包含 {len(validation_df)} 条异常记录，保存到 {validation_file}")
        
        # 为每个电池ID创建单独的工作表，方便分配给不同工程师
        for bid in validation_df['stringid'].unique():
            battery_validation = validation_df[validation_df['stringid'] == bid].copy()
            battery_file = os.path.join(output_dir, f'validation_battery_{bid}.csv')
            battery_validation.to_csv(battery_file, index=False)
            print(f"为电池 {bid} 创建验证工作表，包含 {len(battery_validation)} 条记录")
    
    # 生成统计报告
    generate_stats(df_original, output_dir)
    
    return df_original

def features_to_analyze():
    """
    返回需要分析的特征列表
    """
    return [
        'systemvolt', 'totalcurrenta', 'soc', 'soh',
        'hcellv', 'lcellv', 'averagecellv', 'cellvdelta',
        'htempc', 'ltempc', 'averagecelltempc', 'tempcdelta'
    ]

def generate_stats(df, output_dir):
    """
    为标记后的数据生成统计数据和可视化
    """
    # 按电池ID统计异常
    battery_stats = df.groupby('stringid')['is_anomaly'].agg(['sum', 'count']).reset_index()
    battery_stats['percentage'] = (battery_stats['sum'] / battery_stats['count'] * 100).round(2)
    battery_stats.columns = ['battery_id', 'anomaly_count', 'total_count', 'anomaly_percentage']
    battery_stats = battery_stats.sort_values('anomaly_percentage', ascending=False)
    
    # 保存电池统计数据
    battery_stats_file = os.path.join(output_dir, 'battery_anomaly_stats.csv')
    battery_stats.to_csv(battery_stats_file, index=False)
    print(f"已保存电池异常统计数据到 {battery_stats_file}")
    
    # 按特征统计异常
    feature_counts = {}
    for features in df[df['is_anomaly']]['anomalous_features']:
        if isinstance(features, str) and features:
            for feature in features.split(','):
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
    
    feature_stats = pd.DataFrame({
        'feature': list(feature_counts.keys()),
        'anomaly_count': list(feature_counts.values())
    }).sort_values('anomaly_count', ascending=False)
    
    # 保存特征统计数据
    feature_stats_file = os.path.join(output_dir, 'feature_anomaly_stats.csv')
    feature_stats.to_csv(feature_stats_file, index=False)
    print(f"已保存特征异常统计数据到 {feature_stats_file}")
    
    # 生成可视化
    print("生成可视化图表...")
    
    # 1. 电池异常率条形图
    plt.figure(figsize=(12, 8))
    plt.bar(battery_stats['battery_id'].astype(str), battery_stats['anomaly_percentage'])
    plt.title('各电池异常率 (%)', fontsize=14)
    plt.xlabel('电池ID', fontsize=12)
    plt.ylabel('异常率 (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'battery_anomaly_rate.png'))
    plt.close()
    
    # 2. 特征异常数量条形图
    if not feature_stats.empty:
        plt.figure(figsize=(12, 8))
        plt.bar(feature_stats['feature'], feature_stats['anomaly_count'])
        plt.title('各特征异常点数量', fontsize=14)
        plt.xlabel('特征', fontsize=12)
        plt.ylabel('异常点数量', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_anomaly_count.png'))
        plt.close()
    
    # 3. 时间序列异常分布图
    df['date'] = df['ts'].dt.date
    daily_anomalies = df.groupby('date')['is_anomaly'].sum().reset_index()
    
    plt.figure(figsize=(14, 6))
    plt.plot(daily_anomalies['date'], daily_anomalies['is_anomaly'], 'b-', linewidth=2)
    plt.title('异常点的时间分布', fontsize=14)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('异常点数量', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'anomaly_time_distribution.png'))
    plt.close()
    
    # 4. 电池和特征的热图（如果有足够的数据）
    if len(feature_counts) > 1 and len(battery_stats) > 1:
        # 创建电池-特征矩阵
        battery_feature_matrix = {}
        for bid in df['stringid'].unique():
            battery_feature_matrix[bid] = {}
            for feature in feature_counts.keys():
                battery_feature_matrix[bid][feature] = 0
        
        # 填充矩阵
        for _, row in df[df['is_anomaly']].iterrows():
            if isinstance(row['anomalous_features'], str) and row['anomalous_features']:
                for feature in row['anomalous_features'].split(','):
                    battery_feature_matrix[row['stringid']][feature] += 1
        
        # 转换为DataFrame
        matrix_df = pd.DataFrame(battery_feature_matrix).transpose()
        
        # 绘制热图
        plt.figure(figsize=(14, 10))
        sns.heatmap(matrix_df, annot=True, fmt='d', cmap='YlGnBu')
        plt.title('电池-特征异常数量矩阵', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'battery_feature_matrix.png'))
        plt.close()

def generate_validation_report(validation_file, output_dir):
    """
    从工程师填写的验证工作表生成验证报告
    注意：此函数应在工程师完成验证后运行
    """
    # 检查验证文件是否存在
    if not os.path.exists(validation_file):
        print(f"错误：找不到验证工作表 {validation_file}")
        return
    
    print(f"从 {validation_file} 读取工程师验证结果...")
    try:
        df_validation = pd.read_csv(validation_file)
    except Exception as e:
        print(f"读取验证工作表时出错: {e}")
        return
    
    # 检查关键列是否存在
    required_cols = ['stringid', 'anomalous_features', 'manually_verified', 'detection_correct']
    missing_cols = [col for col in required_cols if col not in df_validation.columns]
    if missing_cols:
        print(f"错误：验证工作表缺少以下列: {missing_cols}")
        return
    
    # 转换数据类型
    if 'manually_verified' in df_validation.columns:
        df_validation['manually_verified'] = df_validation['manually_verified'].astype(bool)
    
    # 计算验证统计数据
    verified_count = df_validation['manually_verified'].sum()
    if verified_count == 0:
        print("警告：没有发现已验证的记录。工程师可能尚未完成验证工作。")
        return
    
    total_count = len(df_validation)
    verification_rate = (verified_count / total_count) * 100
    
    print(f"已验证记录: {verified_count}/{total_count} ({verification_rate:.2f}%)")
    
    # 分析检测正确率（仅考虑已验证的记录）
    verified_df = df_validation[df_validation['manually_verified']].copy()
    correct_detection = (verified_df['detection_correct'] == '正确').sum()
    incorrect_detection = (verified_df['detection_correct'] == '错误').sum()
    uncertain_detection = verified_df.shape[0] - correct_detection - incorrect_detection
    
    accuracy = (correct_detection / verified_df.shape[0]) * 100 if verified_df.shape[0] > 0 else 0
    
    print(f"检测准确率: {accuracy:.2f}%")
    print(f"- 正确检测: {correct_detection}/{verified_df.shape[0]} ({correct_detection/verified_df.shape[0]*100:.2f}%)")
    print(f"- 错误检测: {incorrect_detection}/{verified_df.shape[0]} ({incorrect_detection/verified_df.shape[0]*100:.2f}%)")
    print(f"- 不确定检测: {uncertain_detection}/{verified_df.shape[0]} ({uncertain_detection/verified_df.shape[0]*100:.2f}%)")
    
    # 按特征分析准确率
    feature_accuracy = {}
    for _, row in verified_df.iterrows():
        if isinstance(row['anomalous_features'], str) and row['anomalous_features']:
            for feature in row['anomalous_features'].split(','):
                if feature not in feature_accuracy:
                    feature_accuracy[feature] = {'correct': 0, 'incorrect': 0, 'uncertain': 0, 'total': 0}
                
                feature_accuracy[feature]['total'] += 1
                
                if row['detection_correct'] == '正确':
                    feature_accuracy[feature]['correct'] += 1
                elif row['detection_correct'] == '错误':
                    feature_accuracy[feature]['incorrect'] += 1
                else:
                    feature_accuracy[feature]['uncertain'] += 1
    
    # 生成特征准确率数据帧
    feature_acc_data = []
    for feature, counts in feature_accuracy.items():
        acc_rate = (counts['correct'] / counts['total']) * 100 if counts['total'] > 0 else 0
        feature_acc_data.append({
            'feature': feature,
            'correct': counts['correct'],
            'incorrect': counts['incorrect'],
            'uncertain': counts['uncertain'],
            'total': counts['total'],
            'accuracy_percentage': acc_rate
        })
    
    feature_acc_df = pd.DataFrame(feature_acc_data)
    feature_acc_df = feature_acc_df.sort_values('accuracy_percentage', ascending=False)
    
    # 保存特征准确率报告
    feature_acc_file = os.path.join(output_dir, 'feature_accuracy_report.csv')
    feature_acc_df.to_csv(feature_acc_file, index=False)
    print(f"已保存特征准确率报告到 {feature_acc_file}")
    
    # 生成可视化
    # 1. 特征准确率条形图
    plt.figure(figsize=(12, 8))
    plt.bar(feature_acc_df['feature'], feature_acc_df['accuracy_percentage'])
    plt.title('各特征检测准确率 (%)', fontsize=14)
    plt.xlabel('特征', fontsize=12)
    plt.ylabel('准确率 (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_accuracy.png'))
    plt.close()
    
    # 2. 异常原因分析（如果有填写）
    if 'anomaly_reason' in df_validation.columns:
        reasons = df_validation['anomaly_reason'].dropna().value_counts()
        if len(reasons) > 0:
            plt.figure(figsize=(12, 8))
            plt.bar(reasons.index, reasons.values)
            plt.title('工程师确认的异常原因分布', fontsize=14)
            plt.xlabel('异常原因', fontsize=12)
            plt.ylabel('数量', fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'anomaly_reasons.png'))
            plt.close()
    
    # 生成综合验证报告
    report = f"""
# 电池异常检测验证报告

## 验证摘要
- 异常记录总数: {total_count}
- 已验证记录数: {verified_count} ({verification_rate:.2f}%)
- 整体检测准确率: {accuracy:.2f}%
  - 正确检测: {correct_detection} ({correct_detection/verified_df.shape[0]*100:.2f}%)
  - 错误检测: {incorrect_detection} ({incorrect_detection/verified_df.shape[0]*100:.2f}%)
  - 不确定检测: {uncertain_detection} ({uncertain_detection/verified_df.shape[0]*100:.2f}%)

## 特征准确率
"""
    
    # 添加特征准确率表格
    for _, row in feature_acc_df.iterrows():
        report += f"- {row['feature']}: {row['accuracy_percentage']:.2f}% ({row['correct']}/{row['total']})\n"
    
    # 保存报告
    report_file = os.path.join(output_dir, 'validation_report.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"已生成综合验证报告：{report_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='电池数据异常标记与工程师验证工具')
    parser.add_argument('--mode', choices=['label', 'report'], default='label',
                        help='运行模式: label(标记数据) 或 report(生成验证报告)')
    parser.add_argument('--validation-file', default=None,
                        help='工程师验证工作表路径(仅在report模式下使用)')
    
    args = parser.parse_args()
    
    if args.mode == 'label':
        # 运行标记过程
        label_battery_anomalies()
        print("标记过程完成!")
    elif args.mode == 'report':
        # 生成验证报告
        validation_file = args.validation_file
        if validation_file is None:
            validation_file = 'nixtla_results/labeled_data/engineer_validation_sheet.csv'
        
        generate_validation_report(validation_file, 'nixtla_results/labeled_data')
        print("验证报告生成完成!")
