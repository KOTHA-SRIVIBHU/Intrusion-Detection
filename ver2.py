"""
Intrusion Detection System - Full Pipeline (Both Feature Sets Combined)
- Network features (19 per row) + Statistical features (18 per row)
- Combined = 37 features per row
- Models: Random Forest, XGBoost, Hybrid (Isolation Forest + Random Forest)
- SHAP explainability (force plot fixed)
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import shap
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ================================
# 1. LOAD ORIGINAL DATA (20k each)
# ================================
print("="*80)
print("STEP 1: Loading 20k rows from each original CSV")
print("="*80)

files = {
    '02-14': 'data/02-14-2018.csv',
    '02-16': 'data/02-16-2018.csv',
    '02-21': 'data/02-21-2018.csv'
}
samples = []
for name, path in files.items():
    print(f"Loading {name}...")
    df = pd.read_csv(path, nrows=20000, low_memory=False)
    print(f"  Shape: {df.shape}")
    samples.append(df)
df_raw = pd.concat(samples, ignore_index=True)
print(f"\nCombined shape: {df_raw.shape}")

# ================================
# 2. DATA CLEANING
# ================================
print("\n" + "="*80)
print("STEP 2: Data Cleaning")
print("="*80)

df_raw = df_raw[df_raw['Label'] != 'Label'].copy()
df_raw = df_raw.replace([np.inf, -np.inf], np.nan)
df_raw = df_raw.fillna(0)

feature_cols_all = [c for c in df_raw.columns if c not in ['Timestamp', 'Label']]
for col in feature_cols_all:
    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0)

if 'Timestamp' in df_raw.columns:
    df_raw = df_raw.drop('Timestamp', axis=1)

print(f"After cleaning: {df_raw.shape}")

# ================================
# 3. LABEL ENCODING
# ================================
print("\n" + "="*80)
print("STEP 3: Label Encoding")
print("="*80)

le = LabelEncoder()
df_raw['Label_encoded'] = le.fit_transform(df_raw['Label'])
print("Label mapping:")
for i, lbl in enumerate(le.classes_):
    print(f"  {i} -> {lbl}")
print("\nClass distribution (imbalanced):")
print(df_raw['Label'].value_counts())

# ================================
# 4. FEATURE SCALING (MinMax)
# ================================
print("\n" + "="*80)
print("STEP 4: Feature Scaling (MinMax)")
print("="*80)

X_all = df_raw[feature_cols_all].astype(float)
y_all = df_raw['Label_encoded']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_all)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols_all)
print(f"Scaled feature matrix: {X_scaled.shape}")

# ================================
# 5. FEATURE EXTRACTION (BOTH ALGORITHMS)
# ================================
print("\n" + "="*80)
print("STEP 5: Feature Extraction")
print("="*80)

# ---- Algorithm 1: Network Features (19) ----
network_list = [
    'Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min',
    'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min',
    'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s',
    'Fwd Header Len', 'Bwd Header Len'
]
available_net = [f for f in network_list if f in X_scaled.columns]
X_network = X_scaled[available_net].copy()
print(f"Algorithm 1 (Network Features): {X_network.shape[1]} features (per row).")

# ---- Algorithm 2: Statistical Features (18) ----
# Compute per row: each row's feature values produce statistics (mean, std, skewness, etc.)
print("Computing statistical features per row (this may take 1-2 minutes)...")
start_stat = time.time()

def per_row_statistical_features(df):
    """For each row, compute statistics of its feature values."""
    X_np = df.values
    n_rows, n_cols = X_np.shape
    stats_list = []
    for i in range(n_rows):
        row = X_np[i]
        mean_v = np.mean(row)
        median_v = np.median(row)
        std_v = np.std(row)
        var_v = np.var(row)
        range_v = np.max(row) - np.min(row)
        iqr_v = np.percentile(row, 75) - np.percentile(row, 25)
        if n_cols > 3:
            skew_v = stats.skew(row)
            kurt_v = stats.kurtosis(row)
        else:
            skew_v = kurt_v = 0.0
        # Entropy (fast)
        hist, _ = np.histogram(row, bins=20)
        hist = hist[hist > 0]
        hist = hist / hist.sum() if hist.sum() > 0 else hist
        entropy_v = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0.0
        # Quantiles
        q_vals = np.percentile(row, [10, 25, 50, 75, 90, 95])
        zero_ratio = (row == 0).sum() / n_cols
        cv_v = std_v / mean_v if mean_v != 0 else 0.0
        stats_list.append([
            mean_v, median_v, std_v, var_v, range_v, iqr_v,
            skew_v, kurt_v, entropy_v, zero_ratio, cv_v,
            q_vals[0], q_vals[1], q_vals[2], q_vals[3], q_vals[4], q_vals[5]
        ])
    cols = ['mean','median','std','variance','range','iqr',
            'skewness','kurtosis','entropy','zero_ratio','cv',
            'q10','q25','q50','q75','q90','q95']
    return pd.DataFrame(stats_list, columns=cols, index=df.index)

X_stat = per_row_statistical_features(X_scaled)
print(f"Algorithm 2 (Statistical Features): {X_stat.shape[1]} features (per row) computed in {time.time()-start_stat:.2f}s.")

# ---- Combine both feature sets (horizontal concatenation) ----
# Each row now has 19 network features + 18 statistical features = 37 features
X_combined = pd.concat([X_network, X_stat], axis=1)
print(f"Combined feature set (Network + Statistical): {X_combined.shape[1]} features per row.")

# Save for reference
X_network.to_csv('network_features.csv', index=False)
X_stat.to_csv('statistical_features.csv', index=False)
X_combined.to_csv('combined_features.csv', index=False)
print("Saved network_features.csv, statistical_features.csv, combined_features.csv")

# Use combined features for training
X = X_combined
y = y_all

# ================================
# 6. TRAIN/VAL/TEST SPLIT (70/15/15)
# ================================
print("\n" + "="*80)
print("STEP 6: Train/Val/Test Split")
print("="*80)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ================================
# 7. BASIC MODELS
# ================================
print("\n" + "="*80)
print("STEP 7: Training Basic Models (on combined features)")
print("="*80)

results = {}

# Random Forest
print("\n[1/2] Random Forest (n_estimators=50)...")
start = time.time()
rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
rf.fit(X_train, y_train)
rf_time = time.time() - start
rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)
results['Random Forest'] = {
    'train_time': rf_time,
    'accuracy': accuracy_score(y_test, rf_pred),
    'precision': precision_score(y_test, rf_pred, average='weighted', zero_division=0),
    'recall': recall_score(y_test, rf_pred, average='weighted'),
    'f1': f1_score(y_test, rf_pred, average='weighted'),
    'auc': roc_auc_score(y_test, rf_proba, multi_class='ovr', average='weighted')
}
print(f"  Accuracy: {results['Random Forest']['accuracy']:.4f}, F1: {results['Random Forest']['f1']:.4f}, Time: {rf_time:.2f}s")

# XGBoost
print("\n[2/2] XGBoost (n_estimators=50)...")
start = time.time()
xgb_model = xgb.XGBClassifier(n_estimators=50, random_state=42, eval_metric='mlogloss', n_jobs=1)
xgb_model.fit(X_train, y_train)
xgb_time = time.time() - start
xgb_pred = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)
results['XGBoost'] = {
    'train_time': xgb_time,
    'accuracy': accuracy_score(y_test, xgb_pred),
    'precision': precision_score(y_test, xgb_pred, average='weighted', zero_division=0),
    'recall': recall_score(y_test, xgb_pred, average='weighted'),
    'f1': f1_score(y_test, xgb_pred, average='weighted'),
    'auc': roc_auc_score(y_test, xgb_proba, multi_class='ovr', average='weighted')
}
print(f"  Accuracy: {results['XGBoost']['accuracy']:.4f}, F1: {results['XGBoost']['f1']:.4f}, Time: {xgb_time:.2f}s")

# ================================
# 8. PROPOSED HYBRID MODEL (Isolation Forest + Random Forest)
# ================================
print("\n" + "="*80)
print("STEP 8: Proposed Hybrid Model (Isolation Forest + Random Forest)")
print("="*80)

benign_train = X_train[y_train == 0]
print(f"Benign samples for Isolation Forest: {benign_train.shape}")

start = time.time()
iso_forest = IsolationForest(contamination='auto', random_state=42, n_jobs=1)
iso_forest.fit(benign_train)
if_time = time.time() - start
print(f"  Isolation Forest training time: {if_time:.2f}s")

test_anomaly_flag = iso_forest.predict(X_test)
test_anomaly = (test_anomaly_flag == -1).astype(int)

print("\nTraining Random Forest (classifier)...")
start = time.time()
rf_exp = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
rf_exp.fit(X_train, y_train)
rf_exp_time = time.time() - start
rf_exp_pred = rf_exp.predict(X_test)
rf_exp_proba = rf_exp.predict_proba(X_test)

hybrid_pred = rf_exp_pred.copy()
for i in np.where(test_anomaly == 1)[0]:
    hybrid_pred[i] = np.argmax(rf_exp_proba[i])

hybrid_acc = accuracy_score(y_test, hybrid_pred)
hybrid_prec = precision_score(y_test, hybrid_pred, average='weighted', zero_division=0)
hybrid_rec = recall_score(y_test, hybrid_pred, average='weighted')
hybrid_f1 = f1_score(y_test, hybrid_pred, average='weighted')

hybrid_proba = rf_exp_proba.copy()
n_classes = hybrid_proba.shape[1]
for i in np.where(test_anomaly == 1)[0]:
    one_hot = np.zeros(n_classes)
    one_hot[hybrid_pred[i]] = 1.0
    hybrid_proba[i] = one_hot
hybrid_auc = roc_auc_score(y_test, hybrid_proba, multi_class='ovr', average='weighted')

results['Hybrid (IF+RF)'] = {
    'train_time': if_time + rf_exp_time,
    'accuracy': hybrid_acc,
    'precision': hybrid_prec,
    'recall': hybrid_rec,
    'f1': hybrid_f1,
    'auc': hybrid_auc
}
print(f"\nHybrid evaluation: Acc={hybrid_acc:.4f}, F1={hybrid_f1:.4f}, AUC={hybrid_auc:.4f}")

# ================================
# 9. COMPARISON TABLE
# ================================
print("\n" + "="*80)
print("STEP 9: Model Comparison")
print("="*80)

comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df[['accuracy', 'precision', 'recall', 'f1', 'auc', 'train_time']]
comparison_df.columns = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC', 'Train Time (s)']
print(comparison_df.round(4))
comparison_df.to_csv('model_comparison_both_features.csv')
print("\nSaved: model_comparison_both_features.csv")

# ================================
# 10. SHAP EXPLAINABILITY (fixed force plot)
# ================================
# ================================
# 10. SHAP EXPLAINABILITY (summary only)
# ================================
print("\n" + "="*80)
print("STEP 10: SHAP Analysis (100 test samples)")
print("="*80)

X_test_sample = X_test.sample(n=min(100, len(X_test)), random_state=42)
explainer = shap.TreeExplainer(rf_exp)
shap_values = explainer.shap_values(X_test_sample)

# Summary bar plot
shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('shap_summary_both_features.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: shap_summary_both_features.png")
print("(Force plot skipped due to dimension issues; summary plot sufficient for explainability.)")

# ================================
# 11. SAVE MODEL AND SCALER FOR PREDICTION
# ================================
import joblib

# Save the Random Forest model (you can also save the hybrid, but RF is fine)
joblib.dump(rf_exp, 'rf_model.pkl')
# Save the MinMaxScaler used on raw features
joblib.dump(scaler, 'minmax_scaler.pkl')
# Save the list of feature columns (original raw feature names)
joblib.dump(feature_cols_all, 'raw_feature_columns.pkl')
# Save the network feature column names
joblib.dump(available_net, 'network_feature_names.pkl')
print("\nSaved model, scaler, and feature columns for future predictions.")


print("\n" + "="*80)
print("All tasks completed! Both feature sets combined and used for training.")
print("="*80)