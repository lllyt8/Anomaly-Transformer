import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
from tqdm import tqdm
import matplotlib.dates as mdates

# Allow showing English in Graph correctly
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
print("Loading original data...")
df_original = pd.read_csv(original_data_file, low_memory=False)
df_original.columns = df_original.columns.str.lower()
df_original['ts'] = pd.to_datetime(df_original['ts'], errors='coerce')

# Get all detection result csv files
anomaly_files = glob.glob(os.path.join(results_dir, "anomalies_*.csv"))
print(f"Found {len(anomaly_files)} detection result csv files")

# Create an empty DataFrame to store all anomaly points
all_anomalies = pd.DataFrame()

# Process all anomaly detection result files
for anomaly_file in tqdm(anomaly_files, desc="Processing anomaly files"):
    try:
        # Parse battery ID and feature name from filename
        filename = os.path.basename(anomaly_file)
        parts = filename.replace("anomalies_", "").replace(".csv", "").split("_")
        
        if len(parts) != 2:
            print(f"Warning: Cannot parse filename {filename}")
            continue
        
        battery_id, feature = parts
        
        # Read anomaly detection results
        anomalies_df = pd.read_csv(anomaly_file)
        
        # Ensure timestamp format is correct
        anomalies_df['ds'] = pd.to_datetime(anomalies_df['ds'])
        
        # Filter anomaly points
        anomaly_points = anomalies_df[anomalies_df['anomaly'] == True].copy()
        
        # Skip if no anomaly points
        if len(anomaly_points) == 0:
            continue
        
        # Add battery ID and feature information
        anomaly_points['battery_id'] = battery_id
        anomaly_points['feature'] = feature
        
        # Add anomaly points to total anomalies DataFrame
        all_anomalies = pd.concat([all_anomalies, anomaly_points])
        
        # Data for specific battery ID and feature
        battery_data = df_original[df_original['stringid'] == int(battery_id)].copy()
        
        # Ensure feature column exists in battery_data
        if feature not in battery_data.columns:
            print(f"Warning: Feature {feature} does not exist in original data, skipping")
            continue
        
        # Sort data - key to solving mesh plot issues
        battery_data = battery_data.sort_values('ts')
        anomalies_df = anomalies_df.sort_values('ds')
        
        # Create anomaly point visualization for each feature
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot original data using scatter plot instead of line plot
        ax.scatter(battery_data['ts'], battery_data[feature], 
                color='blue', s=5, alpha=0.6, label='Original Data')
        
        # Plot TimeGPT predictions
        ax.plot(anomalies_df['ds'], anomalies_df['TimeGPT'], 
            'g-', linewidth=2, alpha=0.7, label='Predicted Values')
        
        # Anomaly points
        ax.scatter(anomaly_points['ds'], anomaly_points['y'], 
                color='red', s=50, label='Anomaly Points')
        
        # Set date format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        plt.title(f'Battery {battery_id} - {feature} Anomaly Points')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(anomaly_analysis_dir, f"anomaly_plot_{battery_id}_{feature}.png"))
        plt.close()
        
    except Exception as e:
        print(f"Error processing file {anomaly_file}: {e}")

# Save all anomaly points
if len(all_anomalies) > 0:
    all_anomalies.to_csv(os.path.join(anomaly_analysis_dir, "all_anomalies.csv"), index=False)
    print(f"Saved {len(all_anomalies)} anomaly points to {os.path.join(anomaly_analysis_dir, 'all_anomalies.csv')}")
    
    # Create cross table of battery ID and features, showing anomaly point counts
    anomaly_counts = all_anomalies.groupby(['battery_id', 'feature']).size().unstack().fillna(0).astype(int)
    anomaly_counts.to_csv(os.path.join(anomaly_analysis_dir, "anomaly_counts.csv"))
    print("Generated anomaly count cross table")
    
    # Create heatmap showing anomaly point distribution
    plt.figure(figsize=(14, 10))
    plt.title("Anomaly Point Count by Battery and Feature")
    sns.heatmap(anomaly_counts, annot=True, fmt="d", cmap="YlGnBu")
    plt.tight_layout()
    plt.savefig(os.path.join(anomaly_analysis_dir, "anomaly_counts_heatmap.png"))
    plt.close()
    
    # Additional statistics - Total anomaly points by battery string
    battery_summary = all_anomalies.groupby('battery_id').size().reset_index(name='Anomaly Count')
    battery_summary.to_csv(os.path.join(anomaly_analysis_dir, "battery_anomaly_summary.csv"), index=False)
    
    # Additional statistics - Total anomaly points by feature
    feature_summary = all_anomalies.groupby('feature').size().reset_index(name='Anomaly Count')
    feature_summary.to_csv(os.path.join(anomaly_analysis_dir, "feature_anomaly_summary.csv"), index=False)
    
    # Create bar plots
    plt.figure(figsize=(12, 6))
    sns.barplot(x='battery_id', y='Anomaly Count', data=battery_summary)
    plt.title('Total Anomaly Points by Battery')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(anomaly_analysis_dir, "battery_anomaly_count.png"))
    plt.close()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='feature', y='Anomaly Count', data=feature_summary)
    plt.title('Total Anomaly Points by Feature')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(anomaly_analysis_dir, "feature_anomaly_count.png"))
    plt.close()
else:
    print("Warning：Didn't find any Anomaly Points!")

print("Finished Analyzing Anomaly Points")
