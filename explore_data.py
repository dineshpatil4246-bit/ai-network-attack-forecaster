import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

def explore_data(data_dir="data", output_dir="outputs"):
    # 16. Create outputs/ directory automatically
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Discover all CSV files inside data/
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return

    total_rows = 0
    all_columns = set()
    label_counts = defaultdict(int)
    
    cols_with_nan = set()
    cols_with_inf = set()
    malformed_header_rows = 0
    
    timestamp_stats = {
        'invalid': 0,
        'before_2018': 0,
        'min': pd.NaT,
        'max': pd.NaT,
        'not_chronological_files': []
    }
    
    numeric_stats = {} # feature -> {'sum': x, 'sum_sq': x, 'count': x, 'min': x, 'max': x}
    representative_rows = defaultdict(list)
    
    for file in csv_files:
        print(f"Processing {file}...")
        try:
            # 2. Chunked reading, 3. encoding_errors="replace", low_memory=False to avoid type warnings
            chunk_iter = pd.read_csv(file, chunksize=100000, encoding_errors="replace", low_memory=False)
            
            last_timestamp = pd.NaT
            is_sorted = True
            
            for chunk_idx, chunk in enumerate(chunk_iter):
                if chunk_idx == 0:
                    all_columns.update(chunk.columns.tolist())
                    expected_cols = chunk.columns.tolist()
                
                # 9. Detect malformed/repeated header rows
                # Assumes repeated headers would have the first and last column names matching exactly
                if len(expected_cols) > 0:
                    header_mask = (chunk.iloc[:, 0].astype(str) == str(expected_cols[0])) & (chunk.iloc[:, -1].astype(str) == str(expected_cols[-1]))
                    num_repeated_headers = header_mask.sum()
                    if num_repeated_headers > 0:
                        malformed_header_rows += num_repeated_headers
                        chunk = chunk[~header_mask]
                
                # 4. Report total rows (accumulate)
                total_rows += len(chunk)
                
                # 6. Analyze the Label column
                if 'Label' in chunk.columns:
                    labels = chunk['Label'].value_counts()
                    for label, count in labels.items():
                        label_counts[label] += count
                        
                        # 14. Print up to 3 representative rows for each unique label
                        current_reps = sum(len(df) for df in representative_rows[label])
                        if current_reps < 3:
                            reps_needed = 3 - current_reps
                            reps = chunk[chunk['Label'] == label].head(reps_needed)
                            representative_rows[label].append(reps.copy())
                
                # 7, 8, 12. Handle numeric features & check for NaN/Inf
                for col in chunk.columns:
                    if col in ['Label', 'Timestamp']:
                        continue
                    
                    numeric_col = pd.to_numeric(chunk[col], errors='coerce')
                    
                    if chunk[col].isna().any() or (numeric_col.isna().any() and not chunk[col].isna().all()):
                        cols_with_nan.add(col)
                        
                    if np.isinf(numeric_col).any():
                        cols_with_inf.add(col)
                        
                    valid_mask = ~numeric_col.isna() & ~np.isinf(numeric_col)
                    valid_data = numeric_col[valid_mask].astype(np.float64)
                    
                    if len(valid_data) > 0:
                        if col not in numeric_stats:
                            numeric_stats[col] = {
                                'sum': 0.0,
                                'sum_sq': 0.0,
                                'count': 0,
                                'min': float('inf'),
                                'max': float('-inf')
                            }
                        
                        numeric_stats[col]['sum'] += valid_data.sum()
                        numeric_stats[col]['sum_sq'] += (valid_data ** 2).sum()
                        numeric_stats[col]['count'] += len(valid_data)
                        numeric_stats[col]['min'] = min(numeric_stats[col]['min'], valid_data.min())
                        numeric_stats[col]['max'] = max(numeric_stats[col]['max'], valid_data.max())
                
                # 10. Parse Timestamp using day-first parsing
                if 'Timestamp' in chunk.columns:
                    ts = pd.to_datetime(chunk['Timestamp'], errors='coerce', dayfirst=True)
                    invalid_count = ts.isna().sum()
                    timestamp_stats['invalid'] += invalid_count
                    
                    valid_ts = ts.dropna()
                    if len(valid_ts) > 0:
                        before_2018 = (valid_ts.dt.year < 2018).sum()
                        timestamp_stats['before_2018'] += before_2018
                        
                        chunk_min = valid_ts.min()
                        chunk_max = valid_ts.max()
                        
                        if pd.isna(timestamp_stats['min']) or chunk_min < timestamp_stats['min']:
                            timestamp_stats['min'] = chunk_min
                        if pd.isna(timestamp_stats['max']) or chunk_max > timestamp_stats['max']:
                            timestamp_stats['max'] = chunk_max
                            
                        # Check whether records are chronologically sorted within each file
                        if is_sorted:
                            if not valid_ts.is_monotonic_increasing:
                                is_sorted = False
                            elif pd.notna(last_timestamp) and valid_ts.iloc[0] < last_timestamp:
                                is_sorted = False
                            
                            last_timestamp = valid_ts.iloc[-1]
            
            if not is_sorted:
                timestamp_stats['not_chronological_files'].append(file)
                
        except Exception as e: # 18. Handle file-reading errors gracefully
            print(f"Error processing {file}: {e}")

    # 19. Print a clear summary at the end
    print("\n" + "="*50)
    print("DATA EXPLORATION SUMMARY")
    print("="*50)
    
    print(f"\n4. Total Rows Across All Files: {total_rows}")
    print(f"5. Total Columns: {len(all_columns)}")
    print(f"   Columns:\n   {', '.join(sorted(all_columns))}")
    
    print("\n6. --- Label Distribution ---")
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    if total_rows > 0:
        for label, count in sorted_labels:
            pct = (count / total_rows) * 100
            print(f"   {label}: {count} ({pct:.4f}%)")
            
        # 11. Create outputs/label_distribution.png showing label counts
        if sorted_labels:
            try:
                labels, counts = zip(*sorted_labels)
                plt.figure(figsize=(12, 8))
                plt.bar(labels, counts)
                plt.xticks(rotation=90)
                plt.yscale('log')
                plt.title('Label Distribution (Log Scale)')
                plt.xlabel('Labels')
                plt.ylabel('Count')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'label_distribution.png'))
                plt.close()
                print(f"\n11. Chart saved to {os.path.join(output_dir, 'label_distribution.png')}")
            except Exception as e:
                print(f"    Failed to create chart: {e}")
    
    print("\n7 & 8 & 9. --- Data Quality Issues ---")
    print(f"   Columns with NaN values ({len(cols_with_nan)}): {', '.join(cols_with_nan) if cols_with_nan else 'None'}")
    print(f"   Columns with Inf values ({len(cols_with_inf)}): {', '.join(cols_with_inf) if cols_with_inf else 'None'}")
    print(f"   Malformed/Repeated Header Rows Found: {malformed_header_rows}")
    
    print("\n10. --- Timestamp Analysis ---")
    print(f"   Invalid Timestamps: {timestamp_stats['invalid']}")
    print(f"   Timestamps Before 2018: {timestamp_stats['before_2018']}")
    print(f"   Minimum Timestamp: {timestamp_stats['min']}")
    print(f"   Maximum Timestamp: {timestamp_stats['max']}")
    if timestamp_stats['not_chronological_files']:
        print(f"   Files not chronologically sorted: {', '.join(timestamp_stats['not_chronological_files'])}")
    else:
        print("   All files are chronologically sorted within themselves.")
        
    # 13. Report the 10 numeric features with the highest variance
    print("\n13. --- Top 10 Highest-Variance Numeric Features ---")
    variances = {}
    for col, stats in numeric_stats.items():
        if stats['count'] > 1:
            mean = stats['sum'] / stats['count']
            var = (stats['sum_sq'] - (stats['sum'] ** 2) / stats['count']) / (stats['count'] - 1)
            variances[col] = {
                'variance': var,
                'mean': mean,
                'min': stats['min'],
                'max': stats['max']
            }
            
    top_10_vars = sorted(variances.items(), key=lambda x: x[1]['variance'], reverse=True)[:10]
    for i, (col, stats) in enumerate(top_10_vars, 1):
        print(f"   {i}. {col}")
        print(f"      Variance: {stats['variance']:.4e}")
        print(f"      Mean: {stats['mean']:.4e}")
        print(f"      Min: {stats['min']}")
        print(f"      Max: {stats['max']}")

    # 14. Print up to 3 representative rows for each unique label
    print("\n14. --- Representative Rows by Label (Up to 3) ---")
    for label, dfs in representative_rows.items():
        print(f"\nLabel: {label}")
        if dfs:
            combined = pd.concat(dfs).head(3)
            print(combined.to_string())

# 17. Include if __name__ == "__main__":
if __name__ == "__main__":
    explore_data()
