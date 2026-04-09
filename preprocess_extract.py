# ============================================================================
# INTRUSION DETECTION SYSTEM - PREPROCESSING & FEATURE EXTRACTION
# CIC-IDS 2018 (02-14, 02-16, 02-21)
# Optimized for speed (sampled statistical features)
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from scipy import stats
import time
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# 1. LOAD DATASETS
# ------------------------------
print("="*80)
print("STEP 1: Loading CIC-IDS 2018 datasets")
print("="*80)

file_paths = {
    '02-14': 'data/02-14-2018.csv',
    '02-16': 'data/02-16-2018.csv',
    '02-21': 'data/02-21-2018.csv'
}

datasets = {}
for name, path in file_paths.items():
    print(f"\nLoading {name} ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Original shape: {df.shape}")
    datasets[name] = df

# ------------------------------
# 2. DATA CLEANING
# ------------------------------
print("\n" + "="*80)
print("STEP 2: Data Cleaning")
print("="*80)

def clean_dataset(df, name):
    print(f"\nCleaning {name}...")
    # Remove stray 'Label' row
    df = df[df['Label'] != 'Label'].copy()
    # Ensure Label is string
    df = df[df['Label'].apply(lambda x: isinstance(x, str))].copy()
    # Replace inf with NaN, then fill NaN with 0
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    # Convert feature columns to numeric
    for col in df.columns:
        if col != 'Label':
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    print(f"  After cleaning shape: {df.shape}")
    return df

for name, df in datasets.items():
    datasets[name] = clean_dataset(df, name)

# ------------------------------
# 3. LABEL ENCODING
# ------------------------------
print("\n" + "="*80)
print("STEP 3: Label Encoding")
print("="*80)

label_encoders = {}
for name, df in datasets.items():
    le = LabelEncoder()
    df['Label_encoded'] = le.fit_transform(df['Label'])
    label_encoders[name] = le
    print(f"\n{name} label mapping:")
    for i, lbl in enumerate(le.classes_):
        print(f"  {i} -> {lbl}")
    print(f"  Class distribution:\n{df['Label'].value_counts()}")

# ------------------------------
# 4. FEATURE SCALING & SELECTION
# ------------------------------
print("\n" + "="*80)
print("STEP 4: Feature Scaling & Selection (Variance Threshold)")
print("="*80)

processed_data = {}

for name, df in datasets.items():
    print(f"\n--- Processing {name} ---")
    drop_cols = ['Timestamp', 'Label', 'Label_encoded']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].astype(float)
    y = df['Label_encoded']
    print(f"  Initial features: {X.shape[1]}")
    
    # MinMax Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    # Variance Threshold (remove constant features)
    selector = VarianceThreshold(threshold=0.0)
    X_selected = selector.fit_transform(X_scaled)
    selected_features = X_scaled.columns[selector.get_support()].tolist()
    print(f"  Features after variance threshold: {len(selected_features)}")
    
    processed_data[name] = {
        'X': pd.DataFrame(X_selected, columns=selected_features),
        'y': y,
        'feature_names': selected_features
    }

# ------------------------------
# 5. TRAIN/VAL/TEST SPLIT & SMOTE
# ------------------------------
print("\n" + "="*80)
print("STEP 5: Train/Val/Test Split & SMOTE Balancing")
print("="*80)

final_splits = {}

for name, data in processed_data.items():
    print(f"\n{name}:")
    X = data['X']
    y = data['y']
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"  Before SMOTE -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Train class distribution:\n{y_train.value_counts().sort_index()}")
    
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    print(f"  After SMOTE -> Train: {X_train_bal.shape}")
    print(f"  Balanced class distribution:\n{pd.Series(y_train_bal).value_counts().sort_index()}")
    
    final_splits[name] = {
        'X_train': X_train_bal, 'y_train': y_train_bal,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'feature_names': data['feature_names']
    }

# ------------------------------
# 6. FEATURE EXTRACTION - ALGORITHM 1 (Network)
# ------------------------------
print("\n" + "="*80)
print("STEP 6: Feature Extraction - Algorithm 1 (Network Flow Features)")
print("="*80)

def extract_network_features(df_features, feature_names):
    network_feature_list = [
        'Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
        'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min',
        'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min',
        'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s',
        'Fwd Header Len', 'Bwd Header Len'
    ]
    available = [f for f in network_feature_list if f in feature_names]
    if len(available) < 19:
        print(f"  Warning: Only {len(available)} of 19 network features available.")
    X_net = df_features[available].copy()
    return X_net, available

for name, splits in final_splits.items():
    print(f"\n{name}:")
    X_train = splits['X_train']
    feature_names = splits['feature_names']
    X_train_net, net_feats = extract_network_features(X_train, feature_names)
    print(f"  Extracted {X_train_net.shape[1]} network features.")
    splits['X_train_net'] = X_train_net
    splits['network_feature_names'] = net_feats

# ------------------------------
# 7. FEATURE EXTRACTION - ALGORITHM 2 (Statistical) - SAMPLED
# ------------------------------
print("\n" + "="*80)
print("STEP 7: Feature Extraction - Algorithm 2 (Statistical Features) - Sampled 10%")
print("="*80)

def extract_statistical_features_sampled(df_features, sample_fraction=0.1):
    # Sample rows
    df_sample = df_features.sample(frac=sample_fraction, random_state=42)
    print(f"  Using {len(df_sample)} rows ({sample_fraction*100:.0f}% of {len(df_features)})")
    
    X_np = df_sample.values.astype(float)
    results = []
    for row in X_np:
        mean_val = row.mean()
        median_val = np.median(row)
        std_val = row.std()
        var_val = row.var()
        range_val = row.max() - row.min()
        iqr_val = np.percentile(row, 75) - np.percentile(row, 25)
        if len(row) > 3:
            skew_val = stats.skew(row)
            kurt_val = stats.kurtosis(row)
        else:
            skew_val = kurt_val = 0.0
        hist, _ = np.histogram(row, bins=20)
        hist = hist[hist > 0]
        hist = hist / hist.sum() if hist.sum() > 0 else hist
        entropy_val = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0.0
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        q_vals = np.percentile(row, [q*100 for q in quantiles])
        zero_ratio = (row == 0).sum() / len(row)
        cv_val = std_val / mean_val if mean_val != 0 else 0.0
        
        row_stats = {
            'mean': mean_val, 'median': median_val, 'std': std_val,
            'variance': var_val, 'range': range_val, 'iqr': iqr_val,
            'skewness': skew_val, 'kurtosis': kurt_val, 'entropy': entropy_val,
            'zero_ratio': zero_ratio, 'cv': cv_val,
            'q10': q_vals[0], 'q25': q_vals[1], 'q50': q_vals[2],
            'q75': q_vals[3], 'q90': q_vals[4], 'q95': q_vals[5]
        }
        results.append(row_stats)
    result_df = pd.DataFrame(results, index=df_sample.index)
    return result_df

for name, splits in final_splits.items():
    print(f"\n{name}:")
    X_train = splits['X_train']
    start = time.time()
    X_train_stat = extract_statistical_features_sampled(X_train, sample_fraction=0.1)
    elapsed = time.time() - start
    print(f"  Extracted {X_train_stat.shape[1]} statistical features on {len(X_train_stat)} rows in {elapsed:.2f} sec.")
    splits['X_train_stat'] = X_train_stat
    splits['stat_feature_names'] = X_train_stat.columns.tolist()

# ------------------------------
# 8. SAVE RESULTS
# ------------------------------
print("\n" + "="*80)
print("STEP 8: Saving results")
print("="*80)

for name, splits in final_splits.items():
    # Full balanced training set (original selected features)
    train_df = pd.concat([splits['X_train'], splits['y_train'].rename('Label')], axis=1)
    train_df.to_csv(f"{name}_train_balanced.csv", index=False)
    
    # Network features (full)
    net_df = pd.concat([splits['X_train_net'], splits['y_train'].rename('Label')], axis=1)
    net_df.to_csv(f"{name}_train_network_features.csv", index=False)
    
    # Statistical features (sampled)
    stat_df = pd.concat([splits['X_train_stat'], splits['y_train'].loc[splits['X_train_stat'].index].rename('Label')], axis=1)
    stat_df.to_csv(f"{name}_train_statistical_features_sampled.csv", index=False)
    
    print(f"\n{name}: Saved CSV files.")
    print(f"  Validation set shape: {splits['X_val'].shape}")
    print(f"  Test set shape: {splits['X_test'].shape}")

print("\n" + "="*80)
print("Preprocessing & Feature Extraction Completed Successfully.")
print("="*80)