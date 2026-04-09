"""
Attack Prediction Script - Improved
- Tests with both benign and attack samples
- Prints actual label for comparison
"""

import pandas as pd
import numpy as np
import joblib
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load saved model and preprocessors
rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('minmax_scaler.pkl')
raw_feature_cols = joblib.load('raw_feature_columns.pkl')
network_feature_names = joblib.load('network_feature_names.pkl')

# Label mapping (from ver2.py output)
label_map = {
    0: "Benign",
    1: "DDOS attack-HOIC",
    2: "DDOS attack-LOIC-UDP",
    3: "DoS attacks-SlowHTTPTest",
    4: "FTP-BruteForce"
}

def per_row_statistical_features(row_array):
    row = np.array(row_array).flatten()
    mean_v = np.mean(row)
    median_v = np.median(row)
    std_v = np.std(row)
    var_v = np.var(row)
    range_v = np.max(row) - np.min(row)
    iqr_v = np.percentile(row, 75) - np.percentile(row, 25)
    if len(row) > 3:
        skew_v = stats.skew(row)
        kurt_v = stats.kurtosis(row)
    else:
        skew_v = kurt_v = 0.0
    hist, _ = np.histogram(row, bins=20)
    hist = hist[hist > 0]
    hist = hist / hist.sum() if hist.sum() > 0 else hist
    entropy_v = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0.0
    q_vals = np.percentile(row, [10, 25, 50, 75, 90, 95])
    zero_ratio = (row == 0).sum() / len(row)
    cv_v = std_v / mean_v if mean_v != 0 else 0.0
    stats_list = [
        mean_v, median_v, std_v, var_v, range_v, iqr_v,
        skew_v, kurt_v, entropy_v, zero_ratio, cv_v,
        q_vals[0], q_vals[1], q_vals[2], q_vals[3], q_vals[4], q_vals[5]
    ]
    return np.array(stats_list)

def predict_attack(raw_features_tuple):
    raw = np.array(raw_features_tuple).reshape(1, -1)
    scaled = scaler.transform(raw)
    scaled_df = pd.DataFrame(scaled, columns=raw_feature_cols)
    network_features = scaled_df[network_feature_names].values.flatten()
    stat_features = per_row_statistical_features(scaled.flatten())
    combined = np.concatenate([network_features, stat_features]).reshape(1, -1)
    pred_class = rf_model.predict(combined)[0]
    pred_proba = rf_model.predict_proba(combined)[0]
    attack_name = label_map[pred_class]
    confidence = pred_proba[pred_class]
    return attack_name, confidence, pred_proba

# Test with actual samples from one of the CSVs
print("="*80)
print("Testing predictions with actual rows from 02-14-2018.csv")
print("="*80)

# Load the CSV
df_test = pd.read_csv('data/02-14-2018.csv', low_memory=False)
df_test = df_test[df_test['Label'] != 'Label'].copy()

# Get first benign row
benign_rows = df_test[df_test['Label'] == 'Benign']
if len(benign_rows) > 0:
    benign_row = benign_rows.iloc[0]
    # Extract features (excluding Timestamp and Label)
    features = benign_row[raw_feature_cols].values.flatten().tolist()
    actual_label = benign_row['Label']
    pred_label, conf, all_probs = predict_attack(features)
    print(f"\nActual: {actual_label}")
    print(f"Predicted: {pred_label} (confidence: {conf:.4f})")
    print(f"All class probabilities: {dict(zip(label_map.values(), all_probs))}")

# Get first attack row (non-Benign)
attack_rows = df_test[df_test['Label'] != 'Benign']
if len(attack_rows) > 0:
    attack_row = attack_rows.iloc[0]
    features = attack_row[raw_feature_cols].values.flatten().tolist()
    actual_label = attack_row['Label']
    pred_label, conf, all_probs = predict_attack(features)
    print(f"\nActual: {actual_label}")
    print(f"Predicted: {pred_label} (confidence: {conf:.4f})")
    print(f"All class probabilities: {dict(zip(label_map.values(), all_probs))}")
else:
    print("No attack rows found in the sample.")

# Also test with a specific attack type if you know one (e.g., FTP-BruteForce)
ftp_rows = df_test[df_test['Label'] == 'FTP-BruteForce']
if len(ftp_rows) > 0:
    ftp_row = ftp_rows.iloc[0]
    features = ftp_row[raw_feature_cols].values.flatten().tolist()
    actual_label = ftp_row['Label']
    pred_label, conf, all_probs = predict_attack(features)
    print(f"\nActual (FTP-BruteForce): {actual_label}")
    print(f"Predicted: {pred_label} (confidence: {conf:.4f})")
    print(f"All class probabilities: {dict(zip(label_map.values(), all_probs))}")

print("\n" + "="*80)
print("If predictions are still Benign for attacks, the model may have been saved incorrectly.")
print("Ensure that rf_model.pkl was saved from ver2.py after training on combined features.")
print("="*80)