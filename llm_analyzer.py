import requests
import json
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

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


# Feature rules for battery systems - domain knowledge
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

# Example integration in Streamlit app:
"""
# In your main Streamlit app:
if 'results_df' in locals() and len(results_df) > 0:
    add_llm_analysis_to_streamlit(st, results_df, threshold, api_key=None)
"""
