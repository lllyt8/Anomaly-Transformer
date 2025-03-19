import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class BatterySegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train", target_string_id=None, silent=False):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        
        # Load data
        data = pd.read_csv(data_path)
        
        # Sort data by timestamp
        if 'ts' in data.columns:
            data = data.sort_values('ts')
        
        # Obtain all unique stringId
        all_string_ids = data['stringId'].unique()
        if not silent:
            print(f"Found {len(all_string_ids)} different battery stringIds")
        
        # Process target_string_id
        if target_string_id is not None:
            if target_string_id in all_string_ids:
                string_ids_to_process = [target_string_id]
                if not silent:
                    print(f"Only process String{target_string_id}'s data")
            else:
                raise ValueError(f"The target_string_id {target_string_id} does not exist in the dataset.")
        else:
            string_ids_to_process = all_string_ids
            if not silent:
                print("Processing all battery strings")
        
        # Select features
        selected_features = [
            'systemVolt', 'totalCurrentA', 'soc', 'soh', 
            'hCellV', 'lCellV', 'averageCellV', 'CellVDelta',
            'hTempC', 'lTempC', 'averageCellTempC', 'TempCDelta'
        ]
        
        features_to_use = [f for f in selected_features if f in data.columns]
        if not silent:
            print(f"Using features: {features_to_use}")
        
        # Process data by stringId
        self.string_data = {}
        self.string_scalers = {}
        
        for string_id in string_ids_to_process:
            string_data = data[data['stringId'] == string_id]
            string_features = string_data[features_to_use].values
            string_features = np.nan_to_num(string_features)
            
            scaler = StandardScaler()
            scaler.fit(string_features)
            normalized_data = scaler.transform(string_features)
            
            self.string_scalers[string_id] = scaler
            
            # Split data
            train_size = int(len(normalized_data) * 0.7)
            val_size = int(len(normalized_data) * 0.1)
            
            self.string_data[string_id] = {
                'train': normalized_data[:train_size],
                'val': normalized_data[train_size:train_size+val_size],
                'test': normalized_data[train_size+val_size:]
            }
            
            if not silent:
                print(f"Battery string {string_id} data splits: " +
                      f"train={len(self.string_data[string_id]['train'])}, " +
                      f"val={len(self.string_data[string_id]['val'])}, " +
                      f"test={len(self.string_data[string_id]['test'])}")
        
        # Combine data for all strings
        self.all_data = {
            'train': [],
            'val': [],
            'test': [],
            'train_string_ids': [],
            'val_string_ids': [],
            'test_string_ids': []
        }
        
        for string_id in self.string_data:
            # Generate windows
            for i in range(0, len(self.string_data[string_id]['train']) - win_size + 1, step):
                self.all_data['train'].append(self.string_data[string_id]['train'][i:i + win_size])
                self.all_data['train_string_ids'].append(string_id)
            
            for i in range(0, len(self.string_data[string_id]['val']) - win_size + 1, step):
                self.all_data['val'].append(self.string_data[string_id]['val'][i:i + win_size])
                self.all_data['val_string_ids'].append(string_id)
            
            for i in range(0, len(self.string_data[string_id]['test']) - win_size + 1, step):
                self.all_data['test'].append(self.string_data[string_id]['test'][i:i + win_size])
                self.all_data['test_string_ids'].append(string_id)
        
        # Convert to numpy arrays
        for key in ['train', 'val', 'test']:
            self.all_data[key] = np.array(self.all_data[key])
        
        if not silent:
            print(f"Total windows: train={len(self.all_data['train'])}, " +
                  f"val={len(self.all_data['val'])}, test={len(self.all_data['test'])}")

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

def get_battery_loader(data_path, batch_size, win_size=100, step=1, mode='train', target_string_id=None, silent=False):
    dataset = BatterySegLoader(data_path, win_size, step, mode, target_string_id, silent)
    shuffle = mode == 'train'
    data_loader = DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=0)
    return data_loader
