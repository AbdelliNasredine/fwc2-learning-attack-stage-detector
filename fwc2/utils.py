import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DEFAULT_BASE_DIR = "../datasets"

def load(dataset_name: str, base_dir: str, testing = False) -> pd.DataFrame:
    global DEFAULT_BASE_DIR
    if base_dir:
        DEFAULT_BASE_DIR = base_dir
    base_dir = os.path.join(DEFAULT_BASE_DIR, dataset_name)
    
    print(f'base_dir = {base_dir}')
    
    assert os.path.exists(base_dir), f'{dataset_name} folder must exist'
        
    csv_files = [f for f in os.listdir(base_dir) if f.endswith('.csv')]
    print(csv_files, len(csv_files))
    assert len(csv_files) > 0, f'{dataset_name} must conatin datsets files (csv)'
    
    dfs = []
    for csv_file in csv_files:
        file_path = os.path.join(base_dir, csv_file)
        try:
            df = pd.read_csv(file_path)
            print(f'{csv_file} shape of data = {df.shape}')
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {str(e)}")
            continue
    
    df = pd.concat(dfs, ignore_index=True)    
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    if dataset_name == 'dapt20':
        df['label'] = df['stage']
        df = df.drop(['stage', 'activity'], axis='columns')
    
    return df

def load_pretraining(subsets: list, only_normal: bool, train_ratio: float, random_seed: int) -> tuple:
    dfs = [load(subset) for subset in subsets]
    df = dfs[0] if len(dfs) == 1 else pd.concat(dfs, ignore_index=True)
    
    df = preprocess(df)
    
    if only_normal:
        benign_mask = df.label == 'benign'
        benign_data = df[benign_mask]
        attk_data = df[~benign_mask]
        
        train_mask = np.random.rand(len(benign_data)) < train_ratio
        
        train, test = benign_data[train_mask], pd.concat([benign_data[~train_mask], attk_data], ignore_index=True)
        train_x, train_y = train.drop(['label'], axis=1), train['label']
        test_x, test_y = test.drop(['label'], axis=1), test['label']
        
        return train_x, train_y, test_x, test_y
    else:
        X, y = df.drop('label', axis=1), df['label']
        train_x, test_x, train_y, test_y = train_test_split(
            X, y, 
            test_size= 1 - train_ratio,
            random_state=random_seed,
            stratify=y 
        )
        
        return train_x, train_y, test_x, test_y
         

def preprocess(df, pretraing=True):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # label col of dapt20
    if  'stage' in df.columns:
        df['label'] = df['stage']
        df = df.drop(['stage', 'activity'], axis='columns')

    # fix issue with cicids17
    if 'fwd_header_length.1' in df.columns:
        df = df.drop(['fwd_header_length.1'], axis='columns')
        
    df.label = df.label.str.lower().replace('normal', 'benign')
    
    cols_to_drop = ['source.name', 'flow_id', 'id', 'src_ip', 'dst_ip', 'source_ip', 'destination_ip', 'timestamp', 'src_port', 'dst_port', 'source_port', 'destination_port', 'protocol']
    drop_cols = [col for col in cols_to_drop if col in df.columns]
    if drop_cols:
        df = df.drop(drop_cols, axis='columns')
    
    # replace inf value with nan
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    if pretraing:
        # remove column with 0 variance
        var = df.var(numeric_only=True)
        cols = var[var == 0].index.values
        df = df.drop(cols, axis='columns')

    print(f'shape = {df.shape}')
    
    return df