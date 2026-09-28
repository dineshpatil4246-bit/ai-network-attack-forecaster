import json, glob, joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLS = [
    'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
    'Fwd Pkt Len Max', 'Bwd Pkt Len Max',
    'Flow Byts/s', 'Flow Pkts/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
    'Fwd IAT Mean', 'Bwd IAT Mean',
    'SYN Flag Cnt', 'ACK Flag Cnt', 'PSH Flag Cnt',
    'RST Flag Cnt', 'FIN Flag Cnt', 'URG Flag Cnt'
]

with open('processed/feature_names.json', 'w') as f:
    json.dump(FEATURE_COLS, f)
print('Updated feature_names.json')

dfs = []
for fp in glob.glob('data/*.csv'):
    print('Reading ' + fp)
    df = pd.read_csv(fp, encoding_errors='replace', low_memory=False)
    df.columns = df.columns.str.strip()
    for col in FEATURE_COLS:
        if col in df.columns:
            df = df[df[col] != col]
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
print('Total rows: ' + str(len(df)))

X = df[[c for c in FEATURE_COLS if c in df.columns]].copy()
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

scaler = MinMaxScaler()
scaler.fit(X)
joblib.dump(scaler, 'processed/scaler.pkl')
print('Done! Now run: py predict.py --csv data/Thursday.csv --k 5')