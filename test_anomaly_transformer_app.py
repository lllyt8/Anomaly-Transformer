import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, confusion_matrix
from PIL import Image
import io

# 导入模型和数据加载器
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.battery_loader import get_battery_loader, BatterySegLoader


# 设置页面标题和配置
st.set_page_config(page_title="Anomaly-Transformer Testing Tool", layout="wide")

# KL散度损失计算
def my_kl_loss(p, q):
    """KL divergence loss calculation"""
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

# 主函数
def main():
    st.title("Anomaly-Transformer Testing Tool")
    st.write("This tool allows you to test the Anomaly-Transformer model on battery data.")
    
    # 创建侧边栏用于配置参数
    with st.sidebar:
        st.header("Model Parameters")
        win_size = st.slider("Window Size", min_value=10, max_value=200, value=100, step=5)
        input_c = st.slider("Input Dimension", min_value=1, max_value=20, value=12, step=1)
        output_c = st.slider("Output Dimension", min_value=1, max_value=20, value=12, step=1)
        batch_size = st.slider("Batch Size", min_value=16, max_value=256, value=64, step=16)
        
        st.header("Testing Parameters")
        anomaly_ratio = st.slider("Anomaly Ratio (%)", min_value=0.01, max_value=10.0, value=0.5, step=0.01)
        temperature = st.slider("Temperature", min_value=1, max_value=100, value=50, step=1)
        
        st.header("Model Selection")
        model_path = st.text_input("Model Path", value="checkpoints/BATTERY_checkpoint.pth")
        
        st.header("Optional Settings")
        target_string_id = st.text_input("Target String ID (Optional)", value="")
        if target_string_id == "":
            target_string_id = None
    
    # 预设你的数据路径
    default_data_path = "/Users/orient/Documents/Projects/GitHub/Anomaly-Transformer/data_factory/raw/DanDLN_Data.csv"
    
    # 文件选择部分 - 可以选择预设路径或上传文件
    st.header("Data Selection")
    data_source = st.radio(
        "Select data source:",
        ["Use default path", "Upload CSV file"]
    )
    
    data_path = None
    
    if data_source == "Use default path":
        st.info(f"Using default data path: {default_data_path}")
        data_path = default_data_path
        # 检查文件是否存在
        if not os.path.exists(data_path):
            st.error(f"File not found at {data_path}. Please check the path.")
            return
    else:
        uploaded_file = st.file_uploader("Upload battery data CSV file", type=["csv"])
        if uploaded_file is not None:
            # 保存上传的文件以便DataLoader读取
            with open("temp_uploaded_data.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            data_path = "temp_uploaded_data.csv"
        else:
            st.warning("Please upload a CSV file.")
            return
    
    # 显示数据预览
    if data_path:
        df = pd.read_csv(data_path)
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # 选择设备
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        st.write(f"Using device: {device}")
        
        # 创建模型实例
        model = AnomalyTransformer(
            win_size=win_size, 
            enc_in=input_c, 
            c_out=output_c, 
            e_layers=3
        )
        
        # 检查模型文件是否存在
        model_exists = os.path.exists(model_path)
        st.write(f"Model file exists: {model_exists}")
        
        # 修改后的模型加载部分
        if model_exists:
            try:
                # 尝试使用weights_only=True参数加载模型
                if hasattr(torch, "__version__") and int(torch.__version__.split('.')[0]) >= 1 and int(torch.__version__.split('.')[1]) >= 6:
                    # PyTorch 1.6+支持weights_only参数
                    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
                else:
                    # 较旧版本的PyTorch
                    model.load_state_dict(torch.load(model_path, map_location=device))
                
                model = model.to(device)
                model.eval()
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {e}")
                st.error("Trying alternative loading method...")
                
                try:
                    # 尝试替代加载方法
                    state_dict = torch.load(model_path, map_location=device)
                    # 如果state_dict不是字典而是整个模型，尝试获取state_dict
                    if not isinstance(state_dict, dict):
                        state_dict = state_dict.state_dict()
                    model.load_state_dict(state_dict)
                    model = model.to(device)
                    model.eval()
                    st.success("Model loaded successfully with alternative method!")
                except Exception as e2:
                    st.error(f"Alternative loading also failed: {e2}")
                    return
        else:
            st.error("Model file not found. Please check the path.")
            return
        
        # 开始测试按钮
        if st.button("Start Testing"):
            with st.spinner("Processing data and running model..."):
                # 步骤1: 加载数据
                st.write("Loading data...")
                try:
                    test_loader = get_battery_loader(
                        data_path, 
                        batch_size=batch_size, 
                        win_size=win_size,
                        mode='test', 
                        target_string_id=target_string_id
                    )
                    
                    # 也加载训练数据以计算阈值
                    train_loader = get_battery_loader(
                        data_path, 
                        batch_size=batch_size, 
                        win_size=win_size,
                        mode='train', 
                        target_string_id=target_string_id,
                        silent=True
                    )
                except Exception as e:
                    st.error(f"Error loading data: {e}")
                    return
                
                # 步骤2: 计算训练集的能量分布(用于设置阈值)
                progress_bar = st.progress(0)
                st.write("Computing energy distribution on training set...")
                
                train_energy_scores = []
                criterion = torch.nn.MSELoss(reduce=False)
                
                try:
                    for i, (input_data, _) in enumerate(train_loader):
                        # 更新进度条，防止除以零错误
                        if len(train_loader) > 0:
                            progress_bar.progress(min(1.0, (i + 1) / len(train_loader)))
                        
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
                                                                                                                  win_size)).detach()) * temperature
                                    prior_loss = my_kl_loss(
                                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                                win_size)),
                                        series[u].detach()) * temperature
                                else:
                                    series_loss += my_kl_loss(series[u], (
                                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                                  win_size)).detach()) * temperature
                                    prior_loss += my_kl_loss(
                                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                                win_size)),
                                        series[u].detach()) * temperature
                            
                            # 计算异常分数
                            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                            scores = metric * loss
                            train_energy_scores.append(scores.detach().cpu().numpy())
                except Exception as e:
                    st.error(f"Error during training energy computation: {e}")
                    return
                
                # 处理空结果的情况
                if len(train_energy_scores) == 0:
                    st.error("No training data available for processing.")
                    return
                
                train_energy_scores = np.concatenate(train_energy_scores, axis=0).reshape(-1)
                
                # 步骤3: 在测试集上评估
                progress_bar = st.progress(0)
                st.write("Evaluating on test set...")
                
                test_energy_scores = []
                test_reconstruction_losses = []
                test_labels = []
                sample_indices = []
                string_ids = []
                attention_maps = []
                
                try:
                    for i, (input_data, labels) in enumerate(test_loader):
                        # 更新进度条，防止除以零错误
                        if len(test_loader) > 0:
                            progress_bar.progress(min(1.0, (i + 1) / len(test_loader)))
                        
                        input_data = input_data.float().to(device)
                        
                        with torch.no_grad():
                            output, series, prior, _ = model(input_data)
                            
                            # 保存第一个批次的注意力图
                            if i == 0:
                                for layer_idx, s in enumerate(series):
                                    if s.shape[0] > 0:
                                        # 获取第一个样本的注意力图
                                        attn_map = s[0, 0].detach().cpu().numpy()
                                        attention_maps.append((f"Layer {layer_idx+1}", attn_map))
                            
                            # 计算重构损失
                            rec_loss = torch.mean(criterion(input_data, output), dim=-1)
                            test_reconstruction_losses.append(rec_loss.detach().cpu().numpy())
                            
                            # 计算系列损失和先验损失
                            series_loss = 0.0
                            prior_loss = 0.0
                            for u in range(len(prior)):
                                if u == 0:
                                    series_loss = my_kl_loss(series[u], (
                                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                                  win_size)).detach()) * temperature
                                    prior_loss = my_kl_loss(
                                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                                win_size)),
                                        series[u].detach()) * temperature
                                else:
                                    series_loss += my_kl_loss(series[u], (
                                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                                  win_size)).detach()) * temperature
                                    prior_loss += my_kl_loss(
                                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                                win_size)),
                                        series[u].detach()) * temperature
                            
                            # 计算异常分数
                            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                            scores = metric * rec_loss
                            test_energy_scores.append(scores.detach().cpu().numpy())
                            
                            # 保存标签
                            test_labels.append(labels.numpy())
                            
                            # 保存样本索引
                            batch_indices = list(range(i * batch_size, (i + 1) * batch_size))
                            batch_indices = batch_indices[:len(input_data)]  # 处理最后一个批次
                            sample_indices.extend(batch_indices)
                            
                            # 如果数据加载器提供了电池组ID，则记录它们
                            if hasattr(test_loader.dataset, 'all_data') and 'test_string_ids' in test_loader.dataset.all_data:
                                for idx in batch_indices:
                                    if idx < len(test_loader.dataset.all_data['test_string_ids']):
                                        string_ids.append(test_loader.dataset.all_data['test_string_ids'][idx])
                except Exception as e:
                    st.error(f"Error during test evaluation: {e}")
                    return
                
                # 处理空结果的情况
                if len(test_energy_scores) == 0:
                    st.error("No test data available for processing.")
                    return
                
                test_energy_scores = np.concatenate(test_energy_scores, axis=0).reshape(-1)
                test_reconstruction_losses = np.concatenate(test_reconstruction_losses, axis=0).reshape(-1)
                if len(test_labels) > 0:
                    test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
                
                # 步骤4: 计算阈值并生成预测结果
                combined_scores = np.concatenate([train_energy_scores, test_energy_scores])
                threshold = np.percentile(combined_scores, 100 - anomaly_ratio)
                
                predictions = (test_energy_scores > threshold).astype(int)
                
                # 修复：确定所有列的最小长度，避免不同长度的数组
                min_length = min(
                    len(sample_indices),
                    len(test_energy_scores),
                    len(test_reconstruction_losses),
                    len(predictions)
                )
                
                # 创建结果DataFrame，确保所有数组使用相同的长度
                results_data = {
                    'Sample_Index': sample_indices[:min_length],
                    'Anomaly_Score': test_energy_scores[:min_length],
                    'Reconstruction_Loss': test_reconstruction_losses[:min_length],
                    'Predicted_Label': predictions[:min_length]
                }
                
                # 对其他可选列进行同样的处理
                if string_ids:
                    if len(string_ids) >= min_length:
                        results_data['String_ID'] = string_ids[:min_length]
                    else:
                        # 如果string_ids长度不够，用None填充
                        results_data['String_ID'] = string_ids + [None] * (min_length - len(string_ids))
                
                if len(test_labels) > 0:
                    if len(test_labels) >= min_length:
                        results_data['True_Label'] = test_labels[:min_length]
                    else:
                        # 如果test_labels长度不够，用0填充
                        results_data['True_Label'] = np.concatenate([test_labels, np.zeros(min_length - len(test_labels))])
                
                # 现在所有列都有相同的长度，可以安全创建DataFrame
                results_df = pd.DataFrame(results_data)
            
            # 显示测试结果
            st.subheader("Test Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"Anomaly threshold: {threshold:.6f}")
                st.write(f"Total samples tested: {len(results_df)}")
                anomaly_count = results_df['Predicted_Label'].sum()
                st.write(f"Detected anomalies: {anomaly_count} ({anomaly_count/len(results_df)*100:.2f}%)")
            
            with col2:
                st.write(f"Average anomaly score: {results_df['Anomaly_Score'].mean():.6f}")
                st.write(f"Max anomaly score: {results_df['Anomaly_Score'].max():.6f}")
                st.write(f"Min anomaly score: {results_df['Anomaly_Score'].min():.6f}")
            
            # 显示性能指标(如果有真实标签)
            if 'True_Label' in results_df.columns and np.sum(results_df['True_Label']) > 0:
                st.subheader("Performance Metrics")
                precision, recall, f1, _ = precision_recall_fscore_support(
                    results_df['True_Label'], results_df['Predicted_Label'], average='binary')
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Precision", f"{precision:.4f}")
                with col2:
                    st.metric("Recall", f"{recall:.4f}")
                with col3:
                    st.metric("F1 Score", f"{f1:.4f}")
                
                # 混淆矩阵
                cm = confusion_matrix(results_df['True_Label'], results_df['Predicted_Label'])
                st.write("Confusion Matrix:")
                
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
                
                # ROC曲线
                fpr, tpr, _ = roc_curve(results_df['True_Label'], results_df['Anomaly_Score'])
                roc_auc = auc(fpr, tpr)
                
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curve')
                ax.legend(loc='lower right')
                st.pyplot(fig)
            
            # 显示所有结果
            st.subheader("All Test Samples")
            st.dataframe(results_df)
            
            # 只显示异常样本
            st.subheader("Anomaly Samples")
            anomaly_df = results_df[results_df['Predicted_Label'] == 1]
            st.dataframe(anomaly_df)
            
            # 可视化
            st.subheader("Visualizations")
            
            # 异常分数分布
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(results_df['Anomaly_Score'], kde=True, ax=ax)
            ax.axvline(threshold, color='r', linestyle='--', label='Threshold')
            ax.set_title('Anomaly Score Distribution')
            ax.set_xlabel('Anomaly Score')
            ax.set_ylabel('Count')
            ax.legend()
            st.pyplot(fig)
            
            # 异常分数时间序列
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(results_df['Sample_Index'], results_df['Anomaly_Score'])
            ax.axhline(threshold, color='r', linestyle='--', label='Threshold')
            ax.set_title('Anomaly Score over Samples')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Anomaly Score')
            ax.legend()
            st.pyplot(fig)
            
            # 重构损失时间序列
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(results_df['Sample_Index'], results_df['Reconstruction_Loss'])
            ax.set_title('Reconstruction Loss over Samples')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Reconstruction Loss')
            st.pyplot(fig)
            
            # 显示注意力图
            if len(attention_maps) > 0:
                st.subheader("Attention Maps")
                cols = st.columns(min(3, len(attention_maps)))
                
                for i, (layer_name, attn_map) in enumerate(attention_maps):
                    with cols[i % len(cols)]:
                        fig, ax = plt.subplots(figsize=(5, 5))
                        im = ax.imshow(attn_map, cmap='viridis')
                        ax.set_title(f'Attention Map - {layer_name}')
                        fig.colorbar(im, ax=ax)
                        st.pyplot(fig)
            
            # 提供下载结果的选项
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="anomaly_detection_results.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
