import os
import glob
import pandas as pd
import numpy as np
import json
import joblib
import warnings
from sklearn.preprocessing import MinMaxScaler

def build_dataset():
    print("STEP 1 — Load")
    data_dir = 'data'
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}/")
        return
        
    dfs = []
    for file in csv_files:
        print(f"Loading {file}...")
        df_chunk = pd.read_csv(file, encoding_errors='replace', low_memory=False)
        dfs.append(df_chunk)
        
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows loaded: {len(df)}")
    
    print("\nSTEP 2 — Select features")
    FEATURE_COLS = [
      'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
      'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
      'Fwd Packet Length Max', 'Bwd Packet Length Max',
      'Flow Bytes/s', 'Flow Packets/s',
      'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
      'Fwd IAT Mean', 'Bwd IAT Mean',
      'SYN Flag Count', 'ACK Flag Count', 'PSH Flag Count',
      'RST Flag Count', 'FIN Flag Count', 'URG Flag Count'
    ]
    
    # Clean up column names for easier mapping since CIC-IDS has varying names
    df.columns = df.columns.str.strip()
    
    col_mapping = {
        'Tot Fwd Pkts': 'Total Fwd Packets',
        'Tot Bwd Pkts': 'Total Backward Packets',
        'TotLen Fwd Pkts': 'Total Length of Fwd Packets',
        'TotLen Bwd Pkts': 'Total Length of Bwd Packets',
        'Fwd Pkt Len Max': 'Fwd Packet Length Max',
        'Bwd Pkt Len Max': 'Bwd Packet Length Max',
        'Flow Byts/s': 'Flow Bytes/s',
        'Flow Pkts/s': 'Flow Packets/s',
        'SYN Flag Cnt': 'SYN Flag Count',
        'ACK Flag Cnt': 'ACK Flag Count',
        'PSH Flag Cnt': 'PSH Flag Count',
        'RST Flag Cnt': 'RST Flag Count',
        'FIN Flag Cnt': 'FIN Flag Count',
        'URG Flag Cnt': 'URG Flag Count'
    }
    df.rename(columns=col_mapping, inplace=True)
    
    label_col = None
    if 'Label' in df.columns:
        label_col = 'Label'
    else:
        for col in df.columns:
            if col.strip().lower() == 'label':
                label_col = col
                break
                
    if not label_col:
        raise ValueError("Could not find the label column!")

    selected_cols = []
    missing_cols = []
    for col in FEATURE_COLS:
        if col in df.columns:
            selected_cols.append(col)
        else:
            missing_cols.append(col)
            
    if missing_cols:
        warnings.warn(f"Missing features skipped: {missing_cols}")
        
    df = df[selected_cols + [label_col]].copy()
    
    print("\nSTEP 3 — Clean")
    initial_len = len(df)
    df.dropna(subset=[label_col], inplace=True)
    print(f"Dropped {initial_len - len(df)} rows with NaN label.")
    
    # Ensure numeric columns are actually numeric to correctly identify inf and NaNs
    for col in selected_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Replace np.inf and -np.inf with the column's max/min (non-inf) values
    for col in selected_cols:
        is_inf = np.isinf(df[col])
        if is_inf.any():
            valid_vals = df[col][~is_inf & ~df[col].isna()]
            if len(valid_vals) > 0:
                max_val = valid_vals.max()
                min_val = valid_vals.min()
                df[col] = np.where(df[col] == np.inf, max_val, df[col])
                df[col] = np.where(df[col] == -np.inf, min_val, df[col])
            else:
                df[col].replace([np.inf, -np.inf], 0, inplace=True)

    initial_len = len(df)
    df.dropna(inplace=True)
    print(f"Dropped {initial_len - len(df)} rows with NaN features.")
    
    string_cols = df.select_dtypes(include=['object']).columns
    for col in string_cols:
        df[col] = df[col].astype(str).str.strip()
        
    print("\nSTEP 4 — Encode labels")
    os.makedirs('processed', exist_ok=True)
    
    def encode_label(label_str):
        label_upper = str(label_str).upper()
        if 'BENIGN' in label_upper:
            return 0
        elif 'DOS' in label_upper or 'DDOS' in label_upper:
            return 1
        elif any(k in label_upper for k in ['PORTSCAN', 'FTP', 'SSH', 'BRUTE']):
            return 2
        elif 'BOT' in label_upper:
            return 3
        elif any(k in label_upper for k in ['INFILTRAT', 'WEB', 'XSS', 'SQL', 'HEART']):
            return 4
        else:
            return 1
            
    df['encoded_label'] = df[label_col].apply(encode_label)
    
    mapping_dict = {
        '0': 'BENIGN',
        '1': 'DoS/DDoS/Other',
        '2': 'PortScan/BruteForce',
        '3': 'Bot',
        '4': 'Web/Infiltration'
    }
    with open('processed/label_mapping.json', 'w') as f:
        json.dump(mapping_dict, f, indent=4)
        
    print("\nSTEP 5 — Normalize")
    scaler = MinMaxScaler()
    df[selected_cols] = scaler.fit_transform(df[selected_cols])
    joblib.dump(scaler, 'processed/scaler.pkl')
    
    print("\nSTEP 6 — Create sequences (sliding window)")
    window_size = 20
    step_size = 5
    
    X_list = []
    y_list = []
    
    features_array = df[selected_cols].values
    labels_array = df['encoded_label'].values
    
    num_samples = len(df)
    
    for i in range(0, num_samples - window_size + 1, step_size):
        window_features = features_array[i : i + window_size]
        window_labels = labels_array[i : i + window_size]
        
        seq_label = window_labels.max()
        
        X_list.append(window_features)
        y_list.append(seq_label)
        
    X_seq = np.array(X_list)
    y_seq = np.array(y_list)
    
    benign_idx = np.where(y_seq == 0)[0]
    attack_idx = np.where(y_seq > 0)[0]
    
    num_benign = len(benign_idx)
    num_attack = len(attack_idx)
    
    if num_attack > 0 and num_benign > num_attack * 8:
        print(f"Downsampling benign sequences... (Benign: {num_benign}, Attack: {num_attack})")
        np.random.seed(42)
        downsampled_benign_idx = np.random.choice(benign_idx, num_attack * 8, replace=False)
        
        keep_idx = np.concatenate([downsampled_benign_idx, attack_idx])
        keep_idx.sort()
        
        X_seq = X_seq[keep_idx]
        y_seq = y_seq[keep_idx]
        
    elif num_attack == 0:
        print("Warning: No attack sequences found!")
        
    print("\nSTEP 7 — Save and report")
    np.save('processed/X_sequences.npy', X_seq)
    np.save('processed/y_labels.npy', y_seq)
    
    with open('processed/feature_names.json', 'w') as f:
        json.dump(selected_cols, f, indent=4)
        
    print(f"X shape: {X_seq.shape}")
    print(f"y shape: {y_seq.shape}")
    
    unique_labels, counts = np.unique(y_seq, return_counts=True)
    print("Class distribution:")
    for lbl, cnt in zip(unique_labels, counts):
        print(f"  Class {lbl} ({mapping_dict.get(str(lbl), 'Unknown')}): {cnt}")

if __name__ == '__main__':
    build_dataset()
