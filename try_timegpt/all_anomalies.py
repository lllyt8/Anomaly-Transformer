import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from datetime import timedelta

# Define file paths
original_data_file = 'data_factory/raw_data/JNM.csv'
anomalies_file = 'nixtla_results/anomaly_analysis/all_anomalies.csv'
output_file = 'nixtla_results/all_points/complete_anomaly_records_fixed.csv'

# Ensure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

print("Loading original data...")
# Load original data
df_original = pd.read_csv(original_data_file, low_memory=False)
# Convert column names to lowercase
df_original.columns = df_original.columns.str.lower()
# Process timestamps using coerce mode - same as original detection code
df_original['ts'] = pd.to_datetime(df_original['ts'], errors='coerce')
# Remove rows with NaT timestamps
df_original = df_original.dropna(subset=['ts'])

print("Loading anomaly point data...")
# Load anomaly data
df_anomalies = pd.read_csv(anomalies_file)
# Process timestamps using coerce mode
df_anomalies['ds'] = pd.to_datetime(df_anomalies['ds'], errors='coerce')
# Remove rows with NaT timestamps
df_anomalies = df_anomalies.dropna(subset=['ds'])

# Create a dictionary to store anomaly feature information
print("Processing anomaly feature information...")
anomaly_features = {}
for _, row in tqdm(df_anomalies.iterrows(), total=len(df_anomalies), desc="Organizing anomaly features"):
    key = (row['battery_id'], row['ds'])
    if key not in anomaly_features:
        anomaly_features[key] = []
    anomaly_features[key].append(row['feature'])

# Match original data with anomaly data using time tolerance
print("Matching anomaly points with original data...")
matched_records = []
time_tolerance = timedelta(minutes=1)  # Set tolerance to 1 minute

# Process data by battery ID groups
battery_ids = df_anomalies['battery_id'].unique()
for battery_id in tqdm(battery_ids, desc="Processing battery groups"):
    # Extract original and anomaly data for this battery
    battery_original = df_original[df_original['stringid'] == battery_id].copy()
    battery_anomalies = df_anomalies[df_anomalies['battery_id'] == battery_id].copy()
    
    if battery_original.empty:
        print(f"Warning: Battery ID {battery_id} does not exist in original data")
        continue
    
    # Sort timestamps
    battery_original = battery_original.sort_values('ts')
    battery_anomalies = battery_anomalies.sort_values('ds')
    
    # Get unique timestamps for anomaly points
    anomaly_times = battery_anomalies['ds'].unique()
    
    for anomaly_time in anomaly_times:
        # Calculate time differences
        battery_original['time_diff'] = abs(battery_original['ts'] - anomaly_time)
        
        # Find row with minimum time difference
        if not battery_original.empty:
            closest_idx = battery_original['time_diff'].idxmin()
            min_diff = battery_original.loc[closest_idx, 'time_diff']
            
            if min_diff <= time_tolerance:
                record = battery_original.loc[closest_idx].copy()
                
                # Add additional information
                record['anomaly_time'] = anomaly_time
                record['time_diff_seconds'] = min_diff.total_seconds()
                
                # Get anomaly features
                key = (battery_id, anomaly_time)
                if key in anomaly_features:
                    record['anomalous_features'] = ','.join(anomaly_features[key])
                else:
                    record['anomalous_features'] = ''
                
                matched_records.append(record)
            else:
                print(f"Warning: No data found within {time_tolerance} for Battery ID {battery_id} at time {anomaly_time}")

# Convert matching results to DataFrame
if matched_records:
    result_df = pd.DataFrame(matched_records)
    
    # Remove temporary columns
    if 'time_diff' in result_df.columns:
        result_df = result_df.drop(columns=['time_diff'])
    
    # Save results
    result_df.to_csv(output_file, index=False)
    print(f"Saved {len(result_df)} matched anomaly records to {output_file}")
    
    # Calculate statistics
    unique_records = result_df.drop_duplicates(subset=['stringid', 'ts'])
    print(f"Contains {len(unique_records)} unique data points")
    
    # Count anomalies for each feature
    feature_counts = {}
    for features in result_df['anomalous_features']:
        if isinstance(features, str) and features:
            for feature in features.split(','):
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
    
    print("\nFeature anomaly counts:")
    for feature, count in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {count}")
else:
    print("No matching anomaly records found")
