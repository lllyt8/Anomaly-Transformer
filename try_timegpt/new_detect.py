import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os
from tqdm import tqdm
from nixtla import NixtlaClient

# plt.rcParams['font.sans-serif'] = ['SimHei']
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

# Set features to analyze
features_to_analyze = [
    'systemvolt', 'totalcurrenta', 'soc', 'soh',
    'hcellv', 'lcellv', 'averagecellv', 'cellvdelta',
    'htempc', 'ltempc', 'averagecelltempc', 'tempcdelta'
]

# Ensure timestamp column format is correct using coerce mode
df['ts'] = pd.to_datetime(df['ts'], errors='coerce')

# Check for missing timestamps
missing_ts = df['ts'].isna().sum()
if missing_ts > 0:
    print(f"Warning: {missing_ts} rows have unparseable timestamps")
    # Drop these rows
    df = df.dropna(subset=['ts'])

# Confirm these features exist in the dataset
available_features = [f for f in features_to_analyze if f in df.columns]
if len(available_features) < len(features_to_analyze):
    missing_features = set(features_to_analyze) - set(available_features)
    print(f"Warning: The following features are missing in the dataset: {missing_features}")
    features_to_analyze = available_features

# Get all battery string IDs
battery_ids = df['stringid'].unique()
print(f"Found {len(battery_ids)} battery string IDs")

# Create data structure for summary report
summary_results = {
    'battery_id': [],
    'feature': [],
    'total_points': [],
    'anomalies_detected': [],
    'anomaly_percentage': []
}

# Perform anomaly detection for each battery string and feature
for battery_id in tqdm(battery_ids, desc="Processing battery strings"):
    for feature in features_to_analyze:
        try:
            # Select data for specific battery string
            battery_df = df[df['stringid'] == battery_id].copy()
            
            # Check if there are enough data points
            if len(battery_df) < 10:
                print(f"  Warning: Battery string {battery_id} only has {len(battery_df)} data points, skipping")
                continue
            
            # Create new dataframe with only timestamp and current feature
            detection_df = pd.DataFrame()
            detection_df['ds'] = battery_df['ts']
            detection_df['y'] = battery_df[feature]
            
            # Remove missing values
            detection_df = detection_df.dropna()
            
            # Check if there are still enough data points
            if len(detection_df) < 10:
                print(f"  Warning: Battery string {battery_id}'s {feature} feature has too many missing values, skipping")
                continue
            
            # Check for duplicate timestamps
            dup_times = detection_df.duplicated(subset=['ds'], keep=False)
            if dup_times.any():
                print(f"  Info: Battery string {battery_id}'s {feature} feature has {dup_times.sum()} duplicate timestamps, taking mean")
                # Take mean value
                detection_df = detection_df.groupby('ds')['y'].mean().reset_index()
            
            # Sort by timestamp
            detection_df = detection_df.sort_values('ds')
            
            # Resample to ensure consistent time intervals
            # First set ds as index
            detection_df = detection_df.set_index('ds')
            
            # Determine appropriate frequency
            time_diff = pd.Series(detection_df.index).diff().median()
            if time_diff.total_seconds() < 60:
                freq = '1min'
            elif time_diff.total_seconds() < 3600:
                freq = '2min'
            else:
                freq = '1D'
            
            # Resample
            detection_df = detection_df.resample(freq).mean()
            
            # Fill missing values
            detection_df = detection_df.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
            
            # Reset index
            detection_df = detection_df.reset_index()
            
            # Ensure no NaN values
            if detection_df['y'].isna().any():
                print(f"  Warning: Battery string {battery_id}'s {feature} feature has unfillable NaN values, skipping")
                continue
            
            # Detect anomalies
            try:
                # Debug output
                if battery_id == battery_ids[0] and feature == features_to_analyze[0]:
                    print("Detection data sample:")
                    print(detection_df.head())
                    detection_df.head().to_csv(f"{results_dir}/detection_sample_{battery_id}_{feature}.csv", index=False)
                
                # Call TimeGPT's anomaly detection function
                try:
                    # Try with freq parameter
                    anomalies_df = nixtla_client.detect_anomalies(
                        df=detection_df,
                        freq=freq
                    )
                except Exception as freq_error:
                    print(f"  Failed using freq parameter: {str(freq_error)}, trying without freq parameter")
                    # If failed, try without freq parameter
                    anomalies_df = nixtla_client.detect_anomalies(
                        df=detection_df
                    )
                
                # Save anomaly detection results
                if anomalies_df is not None and len(anomalies_df) > 0:
                    anomalies_df.to_csv(f"{results_dir}/anomalies_{battery_id}_{feature}.csv", index=False)
                    
                    # Calculate anomaly percentage
                    anomaly_count = anomalies_df['anomaly'].sum()
                    anomaly_percentage = (anomaly_count / len(anomalies_df)) * 100
                    
                    # Update summary results
                    summary_results['battery_id'].append(battery_id)
                    summary_results['feature'].append(feature)
                    summary_results['total_points'].append(len(anomalies_df))
                    summary_results['anomalies_detected'].append(anomaly_count)
                    summary_results['anomaly_percentage'].append(anomaly_percentage)
                    
                    print(f"  Success: Battery string {battery_id}'s {feature} feature detected {anomaly_count} anomalies, {anomaly_percentage:.2f}%")
                    
                    # Plot results (only for first battery string's each feature)
                    if battery_id == battery_ids[0]:
                        try:
                            plt.figure(figsize=(10, 6))
                            nixtla_client.plot(
                                df=detection_df,
                                fcst_df=anomalies_df,
                                plot_anomalies=True
                            )
                            plt.title(f'Battery String {battery_id} - {feature} Anomaly Detection Results')
                            plt.tight_layout()
                            plt.savefig(f"{results_dir}/plot_{battery_id}_{feature}.png")
                            plt.close()
                        except Exception as plot_error:
                            print(f"  Cannot plot {battery_id}'s {feature} feature chart: {str(plot_error)}")
                    
                else:
                    print(f"  Warning: Battery string {battery_id}'s {feature} feature anomaly detection returned no results")
                    
            except Exception as api_error:
                print(f"  Battery string {battery_id}'s {feature} feature API call failed: {str(api_error)}")
            
        except Exception as e:
            print(f"  Error processing battery string {battery_id}'s {feature} feature: {str(e)}")

# Create and save summary report
summary_df = pd.DataFrame(summary_results)

# Check if summary dataframe has data
print("\nSummary dataframe statistics:")
print(f"- Rows: {len(summary_df)}")
if len(summary_df) == 0:
    print("Warning: No successful anomaly detection results, cannot generate summary report and visualizations")
else:
    # Print more statistics
    print(f"- Battery string count: {summary_df['battery_id'].nunique()}")
    print(f"- Feature count: {summary_df['feature'].nunique()}")
    print(f"- Average anomaly percentage: {summary_df['anomaly_percentage'].mean():.2f}%")
    print(f"- Anomaly percentage range: {summary_df['anomaly_percentage'].min():.2f}% to {summary_df['anomaly_percentage'].max():.2f}%")
    
    # Save summary report
    summary_df.to_csv(f"{results_dir}/anomaly_detection_summary.csv", index=False)
    
    # Generate summary visualizations
    print("\nGenerating summary visualizations...")
    
    # Plot anomaly percentage bar chart by feature
    try:
        feature_summary = summary_df.groupby('feature')['anomaly_percentage'].mean().reset_index()
        
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='feature', y='anomaly_percentage', data=feature_summary, palette='viridis')
        plt.title('Average Anomaly Percentage by Feature (%)', fontsize=14)
        plt.xlabel('Feature', fontsize=12)
        plt.ylabel('Anomaly Percentage (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for i, v in enumerate(feature_summary['anomaly_percentage']):
            ax.text(i, v + 0.5, f"{v:.2f}%", ha='center', fontsize=10)
            
        plt.tight_layout()
        plt.savefig(f"{results_dir}/feature_anomaly_percentage.png")
        plt.close()
    except Exception as e:
        print(f"Error generating feature bar chart: {str(e)}")
    
    # Plot anomaly percentage bar chart by battery string ID
    try:
        battery_summary = summary_df.groupby('battery_id')['anomaly_percentage'].mean().reset_index()
        
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='battery_id', y='anomaly_percentage', data=battery_summary, palette='rocket')
        plt.title('Average Anomaly Percentage by Battery String (%)', fontsize=14)
        plt.xlabel('Battery String ID', fontsize=12)
        plt.ylabel('Anomaly Percentage (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for i, v in enumerate(battery_summary['anomaly_percentage']):
            ax.text(i, v + 0.5, f"{v:.2f}%", ha='center', fontsize=10)
            
        plt.tight_layout()
        plt.savefig(f"{results_dir}/battery_anomaly_percentage.png")
        plt.close()
    except Exception as e:
        print(f"Error generating battery string bar chart: {str(e)}")
    
    # Plot anomaly percentage heatmap by feature and battery string ID
    try:
        pivot_df = summary_df.pivot_table(
            values='anomaly_percentage', 
            index='battery_id', 
            columns='feature',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0)
        plt.title('Anomaly Percentage by Battery String and Feature (%)')
        plt.tight_layout()
        plt.savefig(f"{results_dir}/anomaly_percentage_heatmap.png")
        plt.close()
    except Exception as e:
        print(f"Error generating heatmap: {str(e)}")
