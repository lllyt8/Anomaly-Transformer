import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class BatterySegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train", target_id=None, silent=False, features=None):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        
        # Load data
        data = pd.read_csv(data_path)
        
        # Sort data by timestamp if exists
        if 'ts' in data.columns:
            data = data.sort_values('ts')
        
        # Log available columns for debugging
        if not silent:
            print(f"Available columns: {data.columns.tolist()}")
        
        # Create case-insensitive column mapping
        col_map = {col.lower(): col for col in data.columns}
        
        # Determine which ID column to use (prefer macId over stringId)
        # Check for different case variations
        mac_id_variants = ['macid', 'macId', 'MacId', 'MACID', 'macID', 'mac_id', 'MAC_ID']
        string_id_variants = ['stringid', 'stringId', 'StringId', 'STRINGID', 'stringID', 'string_id', 'STRING_ID']
        
        # Find matching column for macId using case-insensitive matching
        id_column = None
        
        # Try to find macId first
        for variant in mac_id_variants:
            if variant.lower() in col_map:
                id_column = col_map[variant.lower()]
                break
                
        # If no macId found, try stringId
        if id_column is None:
            for variant in string_id_variants:
                if variant.lower() in col_map:
                    id_column = col_map[variant.lower()]
                    break
        
        # If still no ID column found, raise error
        if id_column is None:
            raise ValueError(f"No ID column found. Looked for: {mac_id_variants + string_id_variants}")
            
        if not silent:
            print(f"Using ID column: {id_column}")
            
        # Obtain all unique IDs
        all_ids = data[id_column].unique()
        if not silent:
            print(f"Found {len(all_ids)} different battery {id_column}s")
        
        # Process target_id
        if target_id is not None:
            if target_id in all_ids:
                ids_to_process = [target_id]
                if not silent:
                    print(f"Only process {id_column}={target_id}'s data")
            else:
                raise ValueError(f"The target_{id_column} {target_id} does not exist in the dataset.")
        else:
            ids_to_process = all_ids
            if not silent:
                print(f"Processing all battery {id_column}s")
        
        # Select features - use provided features or default list
        if features is not None:
            selected_features = features
        else:
            selected_features = [
                'systemVolt', 'totalCurrentA', 'soc', 'soh', 
                'hCellV', 'lCellV', 'averageCellV', 'CellVDelta',
                'hTempC', 'lTempC', 'averageCellTempC', 'TempCDelta'
            ]
        
        # Find matching features using case-insensitive comparison
        features_to_use = []
        for feature in selected_features:
            if feature.lower() in col_map:
                features_to_use.append(col_map[feature.lower()])
        
        if not silent:
            print(f"Using features ({len(features_to_use)}): {features_to_use}")
            
        # Ensure we have at least one feature
        if len(features_to_use) == 0:
            raise ValueError("No matching features found in the dataset. Available columns: " + 
                            ", ".join(data.columns.tolist()))
        
        # Process data by ID
        self.device_data = {}
        self.device_scalers = {}
        
        for device_id in ids_to_process:
            device_data = data[data[id_column] == device_id]
            if len(device_data) == 0:
                if not silent:
                    print(f"Warning: No data found for {id_column}={device_id}")
                continue
                
            device_features = device_data[features_to_use].values
            device_features = np.nan_to_num(device_features)
            
            # Check if we have enough data for this device
            if len(device_features) < win_size:
                if not silent:
                    print(f"Warning: Not enough data for {id_column}={device_id}. " +
                          f"Found {len(device_features)} rows, need at least {win_size}.")
                continue
            
            scaler = StandardScaler()
            scaler.fit(device_features)
            normalized_data = scaler.transform(device_features)
            
            self.device_scalers[device_id] = scaler
            
            # Split data
            train_size = int(len(normalized_data) * 0.7)
            val_size = int(len(normalized_data) * 0.1)
            
            # Ensure we have at least some data in each split
            train_size = max(train_size, win_size)
            val_size = max(val_size, win_size)
            if train_size + val_size >= len(normalized_data):
                # If not enough data for separate validation, use part of training
                train_size = int(len(normalized_data) * 0.8)
                val_size = len(normalized_data) - train_size
                
            self.device_data[device_id] = {
                'train': normalized_data[:train_size],
                'val': normalized_data[train_size:train_size+val_size],
                'test': normalized_data[train_size+val_size:]
            }
            
            if not silent:
                print(f"Battery {id_column} {device_id} data splits: " +
                      f"train={len(self.device_data[device_id]['train'])}, " +
                      f"val={len(self.device_data[device_id]['val'])}, " +
                      f"test={len(self.device_data[device_id]['test'])}")
        
        # Check if we have any valid device data
        if len(self.device_data) == 0:
            raise ValueError("No valid data found for any device ID after processing.")
        
        # Combine data for all devices
        self.all_data = {
            'train': [],
            'val': [],
            'test': [],
            'train_device_ids': [],
            'val_device_ids': [],
            'test_device_ids': []
        }
        
        for device_id in self.device_data:
            # Generate windows for training data
            for i in range(0, len(self.device_data[device_id]['train']) - win_size + 1, step):
                self.all_data['train'].append(self.device_data[device_id]['train'][i:i + win_size])
                self.all_data['train_device_ids'].append(device_id)
            
            # Generate windows for validation data
            for i in range(0, len(self.device_data[device_id]['val']) - win_size + 1, step):
                self.all_data['val'].append(self.device_data[device_id]['val'][i:i + win_size])
                self.all_data['val_device_ids'].append(device_id)
            
            # Generate windows for test data
            for i in range(0, len(self.device_data[device_id]['test']) - win_size + 1, step):
                self.all_data['test'].append(self.device_data[device_id]['test'][i:i + win_size])
                self.all_data['test_device_ids'].append(device_id)
        
        # Convert to numpy arrays
        for key in ['train', 'val', 'test']:
            if len(self.all_data[key]) > 0:
                self.all_data[key] = np.array(self.all_data[key])
            else:
                self.all_data[key] = np.array([])
        
        if not silent:
            print(f"Total windows: train={len(self.all_data['train'])}, " +
                  f"val={len(self.all_data['val'])}, test={len(self.all_data['test'])}")
            print(f"Feature dimension: {self.all_data['train'][0].shape if len(self.all_data['train']) > 0 else 'N/A'}")

    def __len__(self):
        if self.mode == "train":
            return len(self.all_data['train'])
        elif self.mode == 'val':
            return len(self.all_data['val'])
        else:  # test
            return len(self.all_data['test'])

    def __getitem__(self, index):
        if self.mode == "train":
            return np.float32(self.all_data['train'][index])
        elif self.mode == 'val':
            return np.float32(self.all_data['val'][index])
        else:  # test
            return np.float32(self.all_data['test'][index])

def get_battery_loader(data_path, batch_size, win_size=100, step=1, mode='train', target_id=None, silent=False, features=None):
    """
    Create a data loader for battery data.
    
    Args:
        data_path (str): Path to the CSV data file
        batch_size (int): Batch size for the data loader
        win_size (int): Window size for time series segments
        step (int): Step size between windows
        mode (str): 'train', 'val', or 'test'
        target_id (str, optional): Specific battery ID to process
        silent (bool): Whether to suppress progress messages
        features (list, optional): List of specific features to use
        
    Returns:
        DataLoader: PyTorch DataLoader for the specified dataset
    """
    dataset = BatterySegLoader(data_path, win_size, step, mode, target_id, silent, features)
    shuffle = mode == 'train'
    data_loader = DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=0)  #  Keep num_workers=0 for CPU training
    return data_loader
