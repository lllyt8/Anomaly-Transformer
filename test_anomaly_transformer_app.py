import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import json
import requests
from datetime import datetime
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, confusion_matrix
from PIL import Image
import io

# 设置页面标题和配置
st.set_page_config(page_title="Anomaly-Transformer Testing Tool", layout="wide")

# BATTERY_FEATURE_RULES - 电池系统领域知识规则
BATTERY_FEATURE_RULES = {
    "systemVolt": "System voltage should remain within ±5% of nominal during normal operation. Rapid drops may indicate internal short circuits.",
    "totalCurrentA": "Current fluctuations above 20% of normal operating levels may indicate load issues or battery degradation.",
    "soc": "State of Charge should decrease gradually during discharge. Sudden drops indicate calculation errors or severe capacity loss.",
    "soh": "State of Health below 80% indicates significant degradation. Rapid decreases are concerning.",
    "hCellV": "Highest cell voltage should remain below upper limit (typically 4.2V for Li-ion). Values approaching this limit indicate overcharge risk.",
    "lCellV": "Lowest cell voltage should stay above lower limit (typically 3.0V for Li-ion). Values approaching this indicate overdischarge risk.",
    "CellVDelta": "Voltage difference between highest and lowest cells. Values above 100mV indicate cell imbalance issues.",
    "hTempC": "Highest temperature should remain below 45°C for most battery types. Higher values indicate cooling issues or internal problems.",
    "lTempC": "Lowest temperature affects charging capabilities. Below 0°C, charging should be limited or avoided for Li-ion batteries.",
    "TempCDelta": "Temperature difference across the battery pack. Values above 5°C indicate airflow or cooling issues."
}


# KL散度损失计算
def my_kl_loss(p, q):
    """
    计算KL散度
    """
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


class LLMAnomalyAnalyzer:
    """Class to analyze anomalies using an LLM API (Deepseek or similar)"""
    
    def __init__(self, api_key=None, api_url=None):
        """
        Initialize the analyzer with API credentials
        
        Args:
            api_key (str): API key for the LLM service
            api_url (str): API endpoint URL
        """
        self.api_key = api_key
        self.api_url = api_url or "https://api.deepseek.com/v1/"  # Default URL
    
    def set_api_key(self, api_key):
        """Set API key after initialization"""
        self.api_key = api_key
    
    def set_api_url(self, api_url):
        """Set API URL after initialization"""
        self.api_url = api_url
    
    def extract_anomaly_context(self, results_df, device_data, threshold, window_size=100):
        """
        Extract context information for detected anomalies
        
        Args:
            results_df (pd.DataFrame): DataFrame with anomaly detection results
            device_data (dict): Dictionary with device data
            threshold (float): Anomaly threshold value
            window_size (int): Size of time windows
            
        Returns:
            dict: Context information for LLM analysis
        """
        # Get anomaly samples
        anomaly_df = results_df[results_df['Predicted_Label'] == 1].copy()
        
        if len(anomaly_df) == 0:
            return {"status": "no_anomalies", "message": "No anomalies detected"}
        
        # Find the anomaly with highest score
        max_anomaly = anomaly_df.loc[anomaly_df['Anomaly_Score'].idxmax()]
        
        # Get device ID if available
        device_id = max_anomaly.get('Device_ID', 'unknown')
        
        # Get feature names
        if 'device_data' in device_data and device_id in device_data.get('device_data', {}):
            # This would require device_data to be passed from the data loader
            feature_names = device_data.get('feature_names', ["Feature_" + str(i) for i in range(window_size)])
        else:
            feature_names = ["Feature_" + str(i) for i in range(window_size)]
        
        # Prepare context information
        context = {
            "status": "success",
            "anomaly_info": {
                "device_id": device_id,
                "anomaly_score": float(max_anomaly['Anomaly_Score']),
                "threshold": float(threshold),
                "score_ratio": float(max_anomaly['Anomaly_Score'] / threshold),
                "total_anomalies": len(anomaly_df),
                "total_samples": len(results_df),
                "anomaly_percentage": float(len(anomaly_df) / len(results_df) * 100)
            },
            "device_stats": {}
        }
        
        # Add device-specific statistics if available
        if 'Device_ID' in results_df.columns:
            device_stats = results_df.groupby('Device_ID')['Predicted_Label'].agg(['count', 'sum'])
            device_stats['anomaly_rate'] = device_stats['sum'] / device_stats['count'] * 100
            
            for idx, row in device_stats.iterrows():
                context["device_stats"][str(idx)] = {
                    "total_samples": int(row['count']),
                    "anomaly_count": int(row['sum']),
                    "anomaly_rate": float(row['anomaly_rate'])
                }
        
        return context
    
    def analyze_anomalies(self, context, feature_rules=None):
        """
        Analyze anomalies using LLM API
        
        Args:
            context (dict): Context information about the anomalies
            feature_rules (dict, optional): Domain-specific rules for features
            
        Returns:
            dict: Analysis results
        """
        if context.get("status") == "no_anomalies":
            return {"status": "no_anomalies", "message": "No anomalies to analyze"}
        
        # Create prompt for LLM
        prompt = self._create_analysis_prompt(context, feature_rules)
        
        # Check if API key is set
        if not self.api_key:
            return {
                "status": "error", 
                "message": "API key not set. Please provide an API key in the settings.",
                "prompt": prompt  # Return prompt for debugging
            }
        
        try:
            # Call LLM API
            response = self._call_llm_api(prompt)
            
            # Process response
            analysis = self._process_llm_response(response)
            analysis["prompt"] = prompt  # Include prompt for debugging
            return analysis
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error calling LLM API: {str(e)}",
                "prompt": prompt
            }
    
    def _create_analysis_prompt(self, context, feature_rules=None):
        """Create prompt for LLM analysis"""
        anomaly_info = context["anomaly_info"]
        device_stats = context["device_stats"]
        
        # Basic prompt
        prompt = f"""You are an expert battery system analyst who specializes in anomaly detection and diagnosis. 
I need your help to analyze some battery anomalies detected by a deep learning model.

# Anomaly Detection Context:
- Device ID with highest anomaly: {anomaly_info['device_id']}
- Anomaly score: {anomaly_info['anomaly_score']:.4f} (threshold: {anomaly_info['threshold']:.4f})
- Score ratio to threshold: {anomaly_info['score_ratio']:.2f}x
- Total anomalies detected: {anomaly_info['total_anomalies']} out of {anomaly_info['total_samples']} samples ({anomaly_info['anomaly_percentage']:.2f}%)

"""
        
        # Add device statistics if available
        if device_stats:
            prompt += "# Device-specific Statistics:\n"
            for device_id, stats in device_stats.items():
                prompt += f"- Device {device_id}: {stats['anomaly_count']} anomalies out of {stats['total_samples']} samples ({stats['anomaly_rate']:.2f}%)\n"
            prompt += "\n"
        
        # Add feature rules if provided
        if feature_rules:
            prompt += "# Feature Domain Knowledge:\n"
            for feature, rule in feature_rules.items():
                prompt += f"- {feature}: {rule}\n"
            prompt += "\n"
        
        # Analysis requests
        prompt += """Based on this information, please provide:

1. Potential causes of the anomalies (list at least 3 possibilities, ordered by likelihood)
2. What type of anomaly pattern this likely represents (e.g., sudden spike, gradual drift, cyclical issue)
3. Recommended next steps for investigation
4. Potential mitigation measures

Your response should be well-structured and written for a technical audience who manages battery systems. 
Focus on actionable insights rather than just theoretical possibilities.
"""
        
        return prompt
    
    def _call_llm_api(self, prompt):
        """Call LLM API with the given prompt"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": "deepseek-chat",  # or appropriate model name
            "messages": [
                {"role": "system", "content": "You are a battery system expert who specializes in anomaly analysis."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,  # Lower temperature for more focused responses
            "max_tokens": 1000
        }
        
        response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        
        return response.json()
    
    def _process_llm_response(self, response):
        """Process the response from LLM API"""
        try:
            # Extract content from response (format depends on the API)
            # This is for Deepseek API format, adjust if using a different LLM
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if not content:
                return {"status": "error", "message": "Empty response from LLM API"}
            
            return {
                "status": "success",
                "analysis": content,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Error processing LLM response: {str(e)}"}


def add_llm_analysis_to_streamlit(st, results_df, threshold, api_key=None):
    """
    Add LLM analysis section to Streamlit application
    
    Args:
        st: Streamlit object
        results_df (pd.DataFrame): DataFrame with anomaly detection results
        threshold (float): Anomaly threshold value
        api_key (str, optional): API key for LLM service
    """
    st.subheader("LLM-Powered Anomaly Analysis")
    
    # API key input
    api_key_input = st.text_input("LLM API Key (Deepseek)", value=api_key or "", type="password")
    
    # Custom API URL (with default)
    with st.expander("Advanced API Settings"):
        api_url = st.text_input("API URL", value="https://api.deepseek.com/v1/chat/completions")
    
    # Initialize analyzer
    analyzer = LLMAnomalyAnalyzer(api_key=api_key_input, api_url=api_url)
    
    # Extract context for analysis
    device_data = {}  # Would be populated with actual data in real implementation
    context = analyzer.extract_anomaly_context(results_df, device_data, threshold)
    
    # Display basic anomaly statistics
    if context["status"] == "no_anomalies":
        st.info("No anomalies detected. LLM analysis is not needed.")
        return
    
    st.write("### Anomaly Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Anomaly Score", f"{context['anomaly_info']['anomaly_score']:.4f}")
    with col2:
        st.metric("Threshold", f"{context['anomaly_info']['threshold']:.4f}")
    with col3:
        st.metric("Score Ratio", f"{context['anomaly_info']['score_ratio']:.2f}x")
    
    # Button to trigger LLM analysis
    if st.button("Analyze with LLM"):
        if not api_key_input:
            st.warning("Please enter an API key to use LLM analysis.")
        else:
            with st.spinner("Analyzing anomalies with LLM..."):
                analysis_result = analyzer.analyze_anomalies(context, BATTERY_FEATURE_RULES)
                
                if analysis_result["status"] == "success":
                    st.write("### Expert Analysis")
                    st.markdown(analysis_result["analysis"])
                    
                    # Save analysis option
                    if st.button("Save Analysis to File"):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"anomaly_analysis_{timestamp}.txt"
                        with open(filename, "w") as f:
                            f.write(f"Anomaly Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                            f.write(f"Device ID: {context['anomaly_info']['device_id']}\n")
                            f.write(f"Anomaly Score: {context['anomaly_info']['anomaly_score']:.4f}\n")
                            f.write(f"Threshold: {context['anomaly_info']['threshold']:.4f}\n\n")
                            f.write(analysis_result["analysis"])
                        
                        st.success(f"Analysis saved to {filename}")
                else:
                    st.error(analysis_result["message"])
                    
                    # Show prompt for debugging
                    with st.expander("Debug Information"):
                        st.text(analysis_result.get("prompt", "Prompt not available"))
    
    # Display example analysis for demo purposes
    with st.expander("Preview Example Analysis"):
        st.markdown("""
        ### Potential Causes of Anomalies

        1. **Cell Imbalance** (Most Likely): The high anomaly score combined with the pattern across multiple devices suggests voltage imbalance between cells. This is typically caused by:
           - Aging cells developing different internal resistance
           - Manufacturing variations becoming more pronounced over time
           - Thermal gradients within the battery pack

        2. **Thermal Management Issues**: The anomaly patterns may indicate cooling system inefficiencies, particularly if:
           - Temperature differentials exceed 5°C across the pack
           - Higher anomaly rates occur during high-demand periods

        3. **BMS Calibration Drift**: Battery Management System calibration may have drifted, causing:
           - Inaccurate SOC calculations
           - Improper cell balancing decisions
           - Misreporting of actual battery conditions

        ### Anomaly Pattern Analysis

        This appears to represent a **gradual drift with periodic spikes** pattern. The baseline anomaly rate suggests underlying degradation, while the higher anomaly scores indicate specific triggering events (possibly high-current discharges or thermal events).

        ### Recommended Investigation Steps

        1. Examine cell voltage data during the highest anomaly score periods
        2. Compare temperature distributions between normal and anomalous periods
        3. Verify BMS calibration against direct measurements
        4. Review usage patterns preceding anomalies
        5. Check for correlation with environmental conditions (temperature, humidity)

        ### Mitigation Measures

        1. **Short-term**:
           - Implement active cell balancing if not already present
           - Reduce maximum discharge rates by 15-20%
           - Increase cooling capacity during high-load operations

        2. **Long-term**:
           - Recalibrate BMS system
           - Consider replacing highest-variance cells
           - Update thermal management strategy
           - Implement predictive maintenance based on anomaly patterns
        """)


# 导入必要的库，这些将在应用启动时导入
def import_model_and_dataloader():
    # 尝试导入模型和数据加载器，如果失败则显示错误
    try:
        from model.AnomalyTransformer import AnomalyTransformer
        from data_factory.battery_loader import get_battery_loader, BatterySegLoader
        return AnomalyTransformer, get_battery_loader
    except ImportError as e:
        st.error(f"Error importing model or data loader: {str(e)}")
        st.info("Please ensure that the model and data_factory modules are in your PYTHONPATH.")
        return None, None


# 主函数
def main():
    st.title("Anomaly-Transformer Testing Tool")
    st.write("This tool allows you to test the Anomaly-Transformer model on battery data.")
    
    # 导入模型和数据加载器
    AnomalyTransformer, get_battery_loader = import_model_and_dataloader()
    if AnomalyTransformer is None or get_battery_loader is None:
        st.stop()
    
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
        target_id = st.text_input("Target ID (Optional)", value="")
        if target_id == "":
            target_id = None
        
        # LLM分析设置
        st.header("LLM Analysis Settings")
        enable_llm = st.checkbox("Enable LLM Analysis", value=True)
        api_key = st.text_input("API Key (Deepseek)", type="password")
    
    # 预设你的数据路径
    default_data_path = st.text_input(
        "Default Data Path", 
        value="data/battery_data.csv",
        help="Enter the default path to your battery data CSV file"
    )
    
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
        try:
            df = pd.read_csv(data_path)
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # 创建特征选择（可选）
            with st.expander("Feature Selection (Optional)"):
                st.write("Select specific features to use. Leave blank to use default features.")
                selected_features = st.multiselect(
                    "Select features",
                    options=df.columns.tolist(),
                    default=[]
                )
                
                # 如果用户没有选择特征，则使用None（让代码使用默认特征）
                features_to_use = selected_features if selected_features else None
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return
        
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
                        target_id=target_id,
                        features=features_to_use
                    )
                    
                    # 也加载训练数据以计算阈值
                    train_loader = get_battery_loader(
                        data_path, 
                        batch_size=batch_size, 
                        win_size=win_size,
                        mode='train', 
                        target_id=target_id,
                        silent=True,
                        features=features_to_use
                    )
                except Exception as e:
                    st.error(f"Error loading data: {e}")
                    st.error(f"Exception details: {str(e.__class__.__name__)}")
                    return
                
                # 步骤2: 计算训练集的能量分布(用于设置阈值)
                progress_bar = st.progress(0)
                st.write("Computing energy distribution on training set...")
                
                train_energy_scores = []
                criterion = torch.nn.MSELoss(reduce=False)
                
                try:
                    for i, input_data in enumerate(train_loader):
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
                    st.error(f"Exception details: {str(e.__class__.__name__)}")
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
                sample_indices = []
                device_ids = []  # 之前是string_ids
                attention_maps = []
                
                try:
                    for i, input_data in enumerate(test_loader):
                        actual_batch_size = input_data.size(0)  # 获取实际的批次大小
                        
                        # 更新进度条
                        if len(test_loader) > 0:
                            progress_bar.progress(min(1.0, (i + 1) / len(test_loader)))
                        
                        input_data = input_data.float().to(device)
                        
                        with torch.no_grad():
                            output, series, prior, sigmas = model(input_data)
                            
                            # Debug information for first batch
                            if i == 0:
                                st.write(f"Batch shapes: input={input_data.shape}, output={output.shape}")
                                st.write(f"Series list length: {len(series)}, Prior list length: {len(prior)}")
                                
                                # Collect attention maps and sigma statistics
                                attention_maps = []
                                for layer_idx, (s, sigma) in enumerate(zip(series, sigmas)):
                                    if s.shape[0] > 0:
                                        # Save attention map
                                        attn_map = s[0, 0].detach().cpu().numpy()
                                        attention_maps.append((f"Layer {layer_idx+1}", attn_map))
                                        
                                        # Log layer statistics
                                        sigma_stats = sigma[0].detach().cpu().numpy()
                                        st.write(f"Layer {layer_idx+1} stats:")
                                        st.write(f"- Attention: min={attn_map.min():.4f}, max={attn_map.max():.4f}, mean={attn_map.mean():.4f}")
                                        st.write(f"- Sigma: min={sigma_stats.min():.4f}, max={sigma_stats.max():.4f}, mean={sigma_stats.mean():.4f}")
                        
                        # Calculate reconstruction loss
                        rec_loss = torch.mean(criterion(input_data, output), dim=-1)
                        test_reconstruction_losses.append(rec_loss.detach().cpu().numpy())
                        
                        # Calculate series and prior losses
                        series_loss = 0.0
                        prior_loss = 0.0
                        for u in range(len(prior)):
                            if u == 0:
                                series_loss = my_kl_loss(series[u], (
                                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, win_size)).detach())
                                prior_loss = my_kl_loss(
                                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, win_size)),
                                    series[u].detach())
                            else:
                                series_loss += my_kl_loss(series[u], (
                                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, win_size)).detach())
                                prior_loss += my_kl_loss(
                                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, win_size)),
                                    series[u].detach())
                        
                        # Calculate anomaly scores
                        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                        scores = metric * rec_loss
                        test_energy_scores.append(scores.detach().cpu().numpy())
                        
                        # Generate correct sample indices
                        start_idx = i * batch_size
                        end_idx = min(start_idx + actual_batch_size, len(test_loader.dataset))
                        batch_indices = list(range(start_idx, end_idx))
                        sample_indices.extend(batch_indices)
                        
                        # Handle device IDs if available
                        if hasattr(test_loader.dataset, 'device_ids'):
                            device_ids.extend([test_loader.dataset.device_ids[idx] for idx in batch_indices[:actual_batch_size]])

                except Exception as e:
                    st.error(f"Error during testing: {e}")
                    st.error(f"Exception details: {str(e.__class__.__name__)}")
                    return

                # After the testing loop, process results
                try:
                    # Check array lengths
                    array_lengths = {
                        'sample_indices': len(sample_indices),
                        'test_energy_scores': len(test_energy_scores),
                        'test_reconstruction_losses': len(test_reconstruction_losses)
                    }
                    if device_ids:
                        array_lengths['device_ids'] = len(device_ids)
                    
                    st.write("Array lengths before alignment:", array_lengths)
                    
                    # Find minimum length and truncate arrays
                    min_length = min(array_lengths.values())
                    st.write(f"Truncating all arrays to length: {min_length}")
                    
                    # Concatenate and reshape arrays
                    test_energy_scores = np.concatenate(test_energy_scores, axis=0).reshape(-1)[:min_length]
                    test_reconstruction_losses = np.concatenate(test_reconstruction_losses, axis=0).reshape(-1)[:min_length]
                    
                    # Calculate threshold and predictions
                    threshold = np.percentile(train_energy_scores, (1 - anomaly_ratio) * 100)
                    predictions = (test_energy_scores > threshold).astype(int)
                    
                    # Create results DataFrame with window information
                    results_data = {
                        'Sample_Index': sample_indices[:min_length],
                        'Window_Start': [idx - win_size + 1 for idx in sample_indices[:min_length]],
                        'Window_End': sample_indices[:min_length],
                        'Anomaly_Score': test_energy_scores[:min_length],
                        'Reconstruction_Loss': test_reconstruction_losses[:min_length],
                        'Predicted_Label': predictions[:min_length]
                    }
                    
                    # Add time information if available
                    if 'df' in locals() and any(col.lower() in ['timestamp', 'date', 'time'] 
                                               for col in df.columns):
                        time_col = next(col for col in df.columns 
                                       if col.lower() in ['timestamp', 'date', 'time'])
                        time_values = df[time_col].values
                        
                        window_start_times = []
                        window_end_times = []
                        
                        for idx in sample_indices[:min_length]:
                            start_idx = max(0, idx - win_size + 1)
                            end_idx = idx
                            
                            if start_idx < len(time_values) and end_idx < len(time_values):
                                window_start_times.append(time_values[start_idx])
                                window_end_times.append(time_values[end_idx])
                            else:
                                window_start_times.append(None)
                                window_end_times.append(None)
                        
                        results_data['Window_Start_Time'] = window_start_times
                        results_data['Window_End_Time'] = window_end_times
                    
                    if device_ids:
                        results_data['Device_ID'] = device_ids[:min_length]
                    
                    results_df = pd.DataFrame(results_data)
                    
                    # Process and visualize results
                    process_and_visualize_results(results_df, df, win_size, features_to_use)

                except Exception as e:
                    st.error(f"Error processing results: {e}")
                    st.error(f"Exception details: {str(e.__class__.__name__)}")
                    return

def process_and_visualize_results(results_df, df, win_size, features_to_use=None):
    """
    Process and visualize anomaly detection results
    
    Args:
        results_df (pd.DataFrame): DataFrame containing anomaly detection results
        df (pd.DataFrame): Original input data
        win_size (int): Size of the sliding window
        features_to_use (list): List of features to visualize
    """
    st.write("### Detection Results")
    st.write(f"Total samples: {len(results_df)}")
    st.write(f"Detected anomalies: {results_df['Predicted_Label'].sum()}")
    st.write(f"Anomaly ratio: {results_df['Predicted_Label'].mean():.2%}")
    
    # Visualize anomaly windows
    if len(results_df[results_df['Predicted_Label'] == 1]) > 0:
        st.subheader("Top Anomaly Windows")
        
        # Get top 5 anomalies
        top_anomalies = results_df[results_df['Predicted_Label'] == 1].sort_values(
            'Anomaly_Score', ascending=False).head(5)
        
        for idx, anomaly in top_anomalies.iterrows():
            st.write(f"#### Anomaly at index {int(anomaly['Sample_Index'])} (Score: {anomaly['Anomaly_Score']:.4f})")
            
            start_idx = int(anomaly['Window_Start'])
            end_idx = int(anomaly['Window_End'])
            
            if 'Window_Start_Time' in anomaly:
                st.write(f"Time range: {anomaly['Window_Start_Time']} to {anomaly['Window_End_Time']}")
            
            # Plot window data if available
            if df is not None and start_idx >= 0 and end_idx < len(df):
                window_data = df.iloc[start_idx:end_idx+1]
                
                # Plot each feature
                features = features_to_use or df.columns[:min(12, len(df.columns))]
                for feature in features:
                    if feature in window_data.columns and pd.api.types.is_numeric_dtype(window_data[feature]):
                        fig, ax = plt.subplots(figsize=(10, 3))
                        window_data[feature].plot(ax=ax)
                        ax.set_title(f"Feature: {feature}")
                        ax.axvspan(window_data.index[-1] - win_size//4, 
                                 window_data.index[-1], 
                                 alpha=0.3, 
                                 color='red', 
                                 label='Potential anomaly region')
                        plt.legend()
                        st.pyplot(fig)
                        plt.close(fig)
        
        # Create anomaly heatmap
        if df is not None:
            create_anomaly_heatmap(results_df, df, win_size)
        
        # Export results
        create_export_button(results_df)

def create_anomaly_heatmap(results_df, df, win_size):
    """Create and display anomaly heatmap"""
    st.subheader("Anomaly Heatmap")
    
    # Get anomaly indices
    anomaly_indices = []
    for idx, anomaly in results_df[results_df['Predicted_Label'] == 1].iterrows():
        start_idx = int(anomaly['Window_Start'])
        end_idx = int(anomaly['Window_End'])
        anomaly_indices.extend(range(start_idx, end_idx+1))
    
    # Create boolean mask
    anomaly_mask = np.zeros(len(df), dtype=bool)
    valid_indices = [i for i in anomaly_indices if 0 <= i < len(df)]
    anomaly_mask[valid_indices] = True
    
    # Select numeric features
    numeric_features = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    
    if len(numeric_features) > 0:
        # Normalize data
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(df[numeric_features])
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(normalized_data.T, cmap='viridis', robust=True)
        
        # Mark anomaly regions
        for start, end in [(s, min(s+win_size, len(df))) 
                          for s in set(anomaly_indices) if s < len(df)]:
            ax.axvspan(start, end, color='red', alpha=0.2)
        
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Features')
        ax.set_title('Feature Values with Anomaly Regions Highlighted')
        ax.set_yticks(range(len(numeric_features)))
        ax.set_yticklabels(numeric_features)
        
        st.pyplot(fig)
        plt.close(fig)

def create_export_button(results_df):
    """Create download button for anomaly results"""
    st.subheader("Export Results")
    
    anomalies_df = results_df[results_df['Predicted_Label'] == 1].copy()
    csv = anomalies_df.to_csv(index=False)
    
    st.download_button(
        label="Download Anomaly Results as CSV",
        data=csv,
        file_name="anomaly_results.csv",
        mime="text/csv",
    )

if __name__ == "__main__":
    main()
