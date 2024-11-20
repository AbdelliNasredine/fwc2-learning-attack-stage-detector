import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


DATA_BASE_DIR = "../datasets"

class FWC2Dataset(Dataset):
    def __init__(self, data, target, columns=None):
        self.data = np.array(data)
        self.target = np.array(target)
        self.columns = columns

    @property
    def features_low(self):
        return self.data.min(axis=0)

    @property
    def features_high(self):
        return self.data.max(axis=0)

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32)

    def __len__(self):
        return len(self.data)


def _load(dataset_name = 'dapt20', only_normal = False):
    base_dir = os.path.join(DATA_BASE_DIR, dataset_name)
    
    assert os.path.exists(base_dir), f'{dataset_name} folder must exist'
        
    csv_files = [f for f in os.listdir(base_dir) if f.endswith('.csv')]
    print(csv_files, len(csv_files))
    assert len(csv_files) > 0, f'{dataset_name} must conatin datsets files (csv)'
    
    dfs = []
    for csv_file in csv_files:
        file_path = os.path.join(base_dir, csv_file)
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {str(e)}")
            continue
    
    df = pd.concat(dfs, ignore_index=True)    
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    if dataset_name == 'dapt20':
        df['label'] = df['stage']
        df = df.drop(['stage', 'activity'], axis='columns')
        
    df.label = df.label.str.lower().replace('normal', 'benign')
    
    
    return df

def load_pretraining(subsets = ['cicidsd17'], train_ratio = 0.7):
    dfs = [_load(subset, only_normal=True) for subset in subsets]
    df = dfs[0] if len(dfs) == 1 else pd.concat(dfs, ignore_index=True)
    
    # fix issue with cicids17
    df = df.drop(['fwd_header_length.1'], axis='columns')
    
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df.label = df.label.str.lower().replace('normal', 'benign')
    
    cols_to_drop = ['source.name', 'flow_id', 'id', 'src_ip', 'dst_ip', 'source_ip', 'destination_ip', 'timestamp', 'src_port', 'dst_port', 'source_port', 'destination_port', 'protocol']
    drop_cols = [col for col in cols_to_drop if col in df.columns]
    if drop_cols:
        df = df.drop(drop_cols, axis='columns')
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    # remove column with 0 variance
    var = df.var(numeric_only=True)
    cols = var[var == 0].index.values
    print(f'zero_var columns = {cols}')
    df = df.drop(cols, axis='columns')
    
    benign_mask = df.label == 'benign'
    benign_data = df[benign_mask]
    attk_data = df[~benign_mask]
    
    train_mask = np.random.rand(len(benign_data)) < train_ratio
    
    # train & test data
    train, test = benign_data[train_mask], pd.concat([benign_data[~train_mask], attk_data])
    train_x, train_y = train.drop(['label'], axis=1), train['label']
    test_x, test_y = test.drop(['label'], axis=1), train['label']
    
    return train_x, train_y, test_x, test_y

class Data:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.base_path = '../datasets'
        
        self.target_col = 'label'
        if dataset_name == 'dapt20':
            self.target_col = 'stage'


    # def load_pretraining(self, subsets = ['cicidsd17'], train_ratio = 0.7):
    #     dfs = [self._load(subset, only_normal=True) for subset in subsets]
    #     df = dfs[0] if len(dfs) == 1 else pd.concat(dfs, ignore_index=True)
        
    #     # fix issue with cicids17
    #     df = df.drop(['fwd_header_length.1'], axis='columns')
        
    #     df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    #     df.label = df.label.str.lower().replace('normal', 'benign')
        
    #     cols_to_drop = ['source.name', 'flow_id', 'id', 'src_ip', 'dst_ip', 'source_ip', 'destination_ip', 'timestamp', 'src_port', 'dst_port', 'source_port', 'destination_port', 'protocol']
    #     drop_cols = [col for col in cols_to_drop if col in df.columns]
    #     if drop_cols:
    #         df = df.drop(drop_cols, axis='columns')
        
    #     df = df.dropna()
        
    #     benign_mask = df.label == 'benign'
    #     benign_data = df[benign_mask]
    #     attk_data = df[~benign_mask]
        
    #     train_mask = np.random.rand(len(benign_data)) < train_ratio
        

        
    def _load(self, dataset_name = 'dapt20', only_normal = False):
        base_dir = os.path.join(self.base_path, dataset_name)
        
        assert os.path.exists(base_dir), f'{self.dataset_name} folder must exist'
            
        csv_files = [f for f in os.listdir(base_dir) if f.endswith('.csv')]
        print(csv_files, len(csv_files))
        assert len(csv_files) > 0, f'{self.dataset_name} must conatin datsets files (csv)'
        
        dfs = []
        for csv_file in csv_files:
            file_path = os.path.join(base_dir, csv_file)
            try:
                df = pd.read_csv(file_path)
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {csv_file}: {str(e)}")
                continue
        
        df = pd.concat(dfs, ignore_index=True)    
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        if self.dataset_name == 'dapt20':
            df['label'] = df['stage']
            df = df.drop(['stage', 'activity'], axis='columns')
            
        df.label = df.label.str.lower().replace('normal', 'benign')
        
        
        return df
    
    def _preprocess(self, df, drop_heighly_correlated_feats = False):
        # remove index, id and ip columns
        cols_to_drop = ['source.name', 'flow_id', 'id', 'src_ip', 'dst_ip', 'source_ip', 'destination_ip', 'timestamp', 'src_port']
        drop_cols = [col for col in cols_to_drop if col in df.columns]
        if drop_cols:
            df = df.drop(drop_cols, axis='columns')
        
        
        # removing n/a items
        df = df.dropna()
        
        # !todo: remove row having zero as protocole value 
        # df = df[df.protocol != 0]
        
        if drop_heighly_correlated_feats:
            # drop heighly correlated features  
            corr_m = df.select_dtypes(include='number').corr().abs()
            upper = corr_m.where(np.triu(np.ones(corr_m.shape), k=1).astype(np.bool_))
            
            hcf = [column for column in upper.columns if any(upper[column] > 0.95)]
            print(f'h_corr_fs = {hcf}')
            
            df = df.drop(hcf, axis=1)
        
        print(f'----- out dim = {df.shape[1]}')
        return df
        
    def load(self):
        df = self._load()
        df = self._preprocess(df)
        
        x = df.drop([self.target_col], axis=1)
        y = df[self.target_col]
        
        return x, y