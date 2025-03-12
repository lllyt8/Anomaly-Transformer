import argparse
import torch
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, confusion_matrix
import seaborn as sns

from model.AnomalyTransformer import AnomalyTransformer
from data_factory.battery_loader import get_battery_loader

def parse_args():
    parser = argparse.ArgumentParser(description="Anomaly Transformer Model Testing")
    # 模型参数
    parser.add_argument('--win_size', type=int, default=100, help="Window size for time series")
    parser.add_argument('--input_c', type=int, default=12, help="Input dimension")
    parser.add_argument('--output_c', type=int, default=12, help="Output dimension")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for testing")
    
    # 指定预训练模型路径
    parser.add_argument('--model_path', type=str, required=True, help="Path to the pretrained model (.pth file)")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the test data (.csv file)")
    parser.add_argument('--anormly_ratio', type=float, default=0.5, help="Ratio of anomalies for threshold setting")
    
    # 可选：指定电池组ID进行测试
    parser.add_argument('--target_string_id', type=str, default=None, help="Target string ID for testing")
    
    # 输出选项
    parser.add_argument('--output_dir', type=str, default='test_results', help="Directory to save test results")
    parser.add_argument('--save_fig', action='store_true', help="Save figures instead of showing")
    
    return parser.parse_args()

def my_kl_loss(p, q):
    """KL divergence loss calculation"""
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

def test_model(args):
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 设备检测
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载测试数据
    print(f"Loading test data from {args.data_path}")
    test_loader = get_battery_loader(
        args.data_path, 
        batch_size=args.batch_size, 
        win_size=args.win_size,
        mode='test', 
        target_string_id=args.target_string_id
    )
    
    # 构建模型并加载预训练权重
    print(f"Loading model from {args.model_path}")
    model = AnomalyTransformer(
        win_size=args.win_size, 
        enc_in=args.input_c, 
        c_out=args.output_c, 
        e_layers=3
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 测试参数
    temperature = 50
    criterion = torch.nn.MSELoss(reduce=False)
    
    # 在训练集上计算能量分布(用于设置阈值)
    print("Computing energy distribution on training set...")
    train_loader = get_battery_loader(
        args.data_path, 
        batch_size=args.batch_size, 
        win_size=args.win_size,
        mode='train', 
        target_string_id=args.target_string_id,
        silent=True
    )
    
    train_energy_scores = []
    
    for i, (input_data, _) in enumerate(train_loader):
        input_data = input_data.float().to(device)
        with torch.no_grad():
            output, series, prior, _ = model(input_data)
            
            # 计算重构损失
            loss = torch.mean(criterion(input_data, output), dim=-1)
            
            # 计算系列损失和先验损失
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   args.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                args.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   args.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                args.win_size)),
                        series[u].detach()) * temperature
            
            # 计算异常分数
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            scores = metric * loss
            train_energy_scores.append(scores.detach().cpu().numpy())
    
    train_energy_scores = np.concatenate(train_energy_scores, axis=0).reshape(-1)
    
    # 在测试集上评估
    print("Evaluating on test set...")
    test_energy_scores = []
    test_losses = []
    test_labels = []
    sample_indices = []
    attention_maps = []
    
    for i, (input_data, labels) in enumerate(test_loader):
        input_data = input_data.float().to(device)
        with torch.no_grad():
            output, series, prior, _ = model(input_data)
            
            # 保存一些注意力图用于可视化
            if i == 0:
                attention_maps = [s[0].detach().cpu().numpy() for s in series]
            
            # 计算重构损失
            loss = torch.mean(criterion(input_data, output), dim=-1)
            rec_loss = loss.detach().cpu().numpy()
            test_losses.append(rec_loss)
            
            # 计算系列损失和先验损失
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   args.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                args.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   args.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                args.win_size)),
                        series[u].detach()) * temperature
            
            # 计算异常分数
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            scores = metric * loss
            test_energy_scores.append(scores.detach().cpu().numpy())
            
            # 保存标签
            test_labels.append(labels.numpy())
            
            # 保存样本索引(如果需要)
            batch_indices = list(range(i * args.batch_size, (i + 1) * args.batch_size))
            batch_indices = batch_indices[:len(input_data)]  # 处理最后一个批次可能不满的情况
            sample_indices.extend(batch_indices)
    
    test_energy_scores = np.concatenate(test_energy_scores, axis=0).reshape(-1)
    test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
    test_losses = np.concatenate(test_losses, axis=0).reshape(-1)
    
    # 计算阈值(使用训练集和测试集的组合)
    combined_scores = np.concatenate([train_energy_scores, test_energy_scores])
    threshold = np.percentile(combined_scores, 100 - args.anormly_ratio)
    print(f"Anomaly threshold: {threshold:.6f}")
    
    # 生成预测结果
    predictions = (test_energy_scores > threshold).astype(int)
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'Sample_Index': sample_indices[:len(test_energy_scores)],
        'Anomaly_Score': test_energy_scores,
        'Reconstruction_Loss': test_losses,
        'Predicted_Label': predictions,
        'True_Label': test_labels[:len(predictions)] if len(test_labels) > 0 else np.zeros_like(predictions)
    })
    
    # 保存结果到CSV
    results_path = os.path.join(args.output_dir, 'test_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # 计算性能指标(如果有真实标签)
    if np.sum(test_labels) > 0:
        precision, recall, f1, _ = precision_recall_fscore_support(test_labels[:len(predictions)], predictions, average='binary')
        fpr, tpr, _ = roc_curve(test_labels[:len(predictions)], test_energy_scores)
        roc_auc = auc(fpr, tpr)
        
        cm = confusion_matrix(test_labels[:len(predictions)], predictions)
        
        # 打印性能指标
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        
        # 保存性能指标
        with open(os.path.join(args.output_dir, 'performance_metrics.txt'), 'w') as f:
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"ROC AUC: {roc_auc:.4f}\n")
            f.write(f"Confusion Matrix:\n{cm}\n")
    
    # 可视化
    visualize_results(results_df, attention_maps, threshold, args)
    
    return results_df, threshold

def visualize_results(results_df, attention_maps, threshold, args):
    """可视化测试结果"""
    plt.figure(figsize=(15, 6))
    
    # 1. 异常分数分布图
    plt.subplot(2, 3, 1)
    sns.histplot(results_df['Anomaly_Score'], kde=True)
    plt.axvline(threshold, color='r', linestyle='--')
    plt.title('Anomaly Score Distribution')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Count')
    
    # 2. 异常分数时间序列
    plt.subplot(2, 3, 2)
    plt.plot(results_df['Sample_Index'], results_df['Anomaly_Score'])
    plt.axhline(threshold, color='r', linestyle='--')
    plt.title('Anomaly Score over Time')
    plt.xlabel('Sample Index')
    plt.ylabel('Anomaly Score')
    
    # 3. 重构损失时间序列
    plt.subplot(2, 3, 3)
    plt.plot(results_df['Sample_Index'], results_df['Reconstruction_Loss'])
    plt.title('Reconstruction Loss over Time')
    plt.xlabel('Sample Index')
    plt.ylabel('Reconstruction Loss')
    
    # 4. 混淆矩阵(如果有真实标签)
    plt.subplot(2, 3, 4)
    if np.sum(results_df['True_Label']) > 0:
        cm = confusion_matrix(results_df['True_Label'], results_df['Predicted_Label'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
    else:
        plt.text(0.5, 0.5, 'No ground truth labels available', 
                horizontalalignment='center', verticalalignment='center')
        plt.title('Confusion Matrix (Unavailable)')
    
    # 5. ROC曲线(如果有真实标签)
    plt.subplot(2, 3, 5)
    if np.sum(results_df['True_Label']) > 0:
        fpr, tpr, _ = roc_curve(results_df['True_Label'], results_df['Anomaly_Score'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
    else:
        plt.text(0.5, 0.5, 'No ground truth labels available', 
                horizontalalignment='center', verticalalignment='center')
        plt.title('ROC Curve (Unavailable)')
    
    plt.tight_layout()
    
    # 保存或显示图表
    if args.save_fig:
        plt.savefig(os.path.join(args.output_dir, 'results_overview.png'), dpi=300)
    else:
        plt.show()
    
    # 可视化注意力图
    if len(attention_maps) > 0:
        fig, axs = plt.subplots(1, len(attention_maps), figsize=(15, 5))
        if len(attention_maps) == 1:
            axs = [axs]
        
        for i, attn_map in enumerate(attention_maps):
            im = axs[i].imshow(attn_map, cmap='viridis')
            axs[i].set_title(f'Attention Map Layer {i+1}')
            fig.colorbar(im, ax=axs[i])
        
        plt.tight_layout()
        
        if args.save_fig:
            plt.savefig(os.path.join(args.output_dir, 'attention_maps.png'), dpi=300)
        else:
            plt.show()

if __name__ == "__main__":
    args = parse_args()
    results_df, threshold = test_model(args)
    
    # 打印结果摘要
    total_samples = len(results_df)
    anomalies = len(results_df[results_df['Predicted_Label'] == 1])
    print(f"\nTest Summary:")
    print(f"Total samples: {total_samples}")
    print(f"Detected anomalies: {anomalies} ({anomalies/total_samples*100:.2f}%)")
    print(f"Anomaly threshold: {threshold:.6f}")
    print(f"Average anomaly score: {results_df['Anomaly_Score'].mean():.6f}")
    print(f"Max anomaly score: {results_df['Anomaly_Score'].max():.6f}")
    
    print("\nTop 10 anomalies by score:")
    top_anomalies = results_df.sort_values('Anomaly_Score', ascending=False).head(10)
    print(top_anomalies[['Sample_Index', 'Anomaly_Score', 'Predicted_Label']])
