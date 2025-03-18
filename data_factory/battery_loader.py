import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

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
        
        # We only process data for the current target_string_id
        if target_string_id is not None:
            if target_string_id in all_string_ids:
                string_ids_to_process = [target_string_id]
                if not silent:
                    print(f"Only process String{target_string_id}'s data")
            else:
                raise ValueError(f"The target_string_id {target_string_id} does not exist in the dataset.")
        else:
            # Or handle all stringIds
            string_ids_to_process = all_string_ids
            if not silent:
                print("处理所有电池组的数据")
        
        # 选择重要特征
        selected_features = [
            'systemVolt', 'totalCurrentA', 'soc', 'soh', 
            'hCellV', 'lCellV', 'averageCellV', 'CellVDelta',
            'hTempC', 'lTempC', 'averageCellTempC', 'TempCDelta'
        ]
        
        # 确保所有选择的特征都在数据中
        features_to_use = [f for f in selected_features if f in data.columns]
        if not silent:
            print(f"使用特征: {features_to_use}")
        
        # 按照stringId分组处理数据
        self.string_data = {}
        self.string_scalers = {}
        
        for string_id in string_ids_to_process:
            # 获取当前电池组的数据
            string_data = data[data['stringId'] == string_id]
            
            # 提取特征数据
            string_features = string_data[features_to_use].values
            
            # 处理缺失值
            string_features = np.nan_to_num(string_features)
            
            # 为每个电池组创建单独的标准化器
            scaler = StandardScaler()
            scaler.fit(string_features)
            normalized_data = scaler.transform(string_features)
            
            # 保存标准化器，测试时需要使用相同的标准化参数
            self.string_scalers[string_id] = scaler
            
            # 划分训练集、验证集和测试集
            train_size = int(len(normalized_data) * 0.7)
            val_size = int(len(normalized_data) * 0.1)
            
            # 存储数据划分
            self.string_data[string_id] = {
                'train': normalized_data[:train_size],
                'val': normalized_data[train_size:train_size+val_size],
                'test': normalized_data[train_size+val_size:],
                'train_labels': np.zeros(train_size),
                'val_labels': np.zeros(val_size),
                'test_labels': np.zeros(len(normalized_data) - train_size - val_size)
            }
            
            if not silent:
                print(f"电池组 {string_id} 数据: train={len(self.string_data[string_id]['train'])}, " +
                      f"val={len(self.string_data[string_id]['val'])}, test={len(self.string_data[string_id]['test'])}")
        
        # 合并所有电池组的数据，用于生成索引
        self.all_data = {
            'train': [],
            'val': [],
            'test': [],
            'train_labels': [],
            'val_labels': [],
            'test_labels': [],
            'train_string_ids': [],
            'val_string_ids': [],
            'test_string_ids': []
        }
        
        for string_id in self.string_data:
            # 生成训练窗口及其对应的电池组ID
            for i in range(0, len(self.string_data[string_id]['train']) - win_size + 1, step):
                self.all_data['train'].append(self.string_data[string_id]['train'][i:i + win_size])
                self.all_data['train_labels'].append(self.string_data[string_id]['train_labels'][i:i + win_size])
                self.all_data['train_string_ids'].append(string_id)
            
            # 生成验证窗口及其对应的电池组ID
            for i in range(0, len(self.string_data[string_id]['val']) - win_size + 1, step):
                self.all_data['val'].append(self.string_data[string_id]['val'][i:i + win_size])
                self.all_data['val_labels'].append(self.string_data[string_id]['val_labels'][i:i + win_size])
                self.all_data['val_string_ids'].append(string_id)
            
            # 生成测试窗口及其对应的电池组ID
            for i in range(0, len(self.string_data[string_id]['test']) - win_size + 1, step):
                self.all_data['test'].append(self.string_data[string_id]['test'][i:i + win_size])
                self.all_data['test_labels'].append(self.string_data[string_id]['test_labels'][i:i + win_size])
                self.all_data['test_string_ids'].append(string_id)
        
        # 转换为numpy数组
        for key in ['train', 'val', 'test', 'train_labels', 'val_labels', 'test_labels']:
            self.all_data[key] = np.array(self.all_data[key])
        
        if not silent:
            print(f"合并后的时间窗口数据: train={len(self.all_data['train'])}, " +
                  f"val={len(self.all_data['val'])}, test={len(self.all_data['test'])}")

    def __len__(self):
        if self.mode == "train":
            return len(self.all_data['train'])
        elif self.mode == 'val':
            return len(self.all_data['val'])
        elif self.mode == 'test':
            return len(self.all_data['test'])
        else:
            return len(self.all_data['test'])

    def __getitem__(self, index):
        if self.mode == "train":
            return np.float32(self.all_data['train'][index]), np.float32(self.all_data['train_labels'][index])
        elif self.mode == 'val':
            return np.float32(self.all_data['val'][index]), np.float32(self.all_data['val_labels'][index])
        elif self.mode == 'test':
            return np.float32(self.all_data['test'][index]), np.float32(self.all_data['test_labels'][index])
        else:
            return np.float32(self.all_data['test'][index]), np.float32(self.all_data['test_labels'][index])

def get_battery_loader(data_path, batch_size, win_size=100, step=1, mode='train', target_string_id=None, silent=False):
    dataset = BatterySegLoader(data_path, win_size, step, mode, target_string_id, silent)
    
    shuffle = False
    if mode == 'train':
        shuffle = True
        
    data_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=0)
    return data_loader
