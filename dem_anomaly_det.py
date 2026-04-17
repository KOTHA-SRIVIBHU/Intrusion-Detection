"""
Refined Hybrid with Intelligent Override + XGBoost Comparison
- Hybrid: Isolation Forest anomaly flag + Random Forest (if flag, override to best attack class)
- Metrics: accuracy, weighted precision/recall/f1, per-class f1, anomaly recall, confusion matrices
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, roc_curve, auc)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ================================
# 1. LOAD DATA AND FEATURE EXTRACTION
# ================================
print("="*80)
print("STEP 1: Loading data and extracting features")
print("="*80)

files = {
    '02-14': 'data/02-14-2018.csv',
    '02-16': 'data/02-16-2018.csv',
    '02-21': 'data/02-21-2018.csv'
}
samples = []
for name, path in files.items():
    df = pd.read_csv(path, nrows=20000, low_memory=False)
    samples.append(df)
df_raw = pd.concat(samples, ignore_index=True)
df_raw = df_raw[df_raw['Label'] != 'Label'].copy()
df_raw = df_raw.replace([np.inf, -np.inf], np.nan).fillna(0)

feature_cols = [c for c in df_raw.columns if c not in ['Timestamp', 'Label']]
for col in feature_cols:
    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0)
if 'Timestamp' in df_raw.columns:
    df_raw = df_raw.drop('Timestamp', axis=1)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df_raw[feature_cols])
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

# Network features
network_list = [
    'Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min',
    'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min',
    'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s',
    'Fwd Header Len', 'Bwd Header Len'
]
available_net = [f for f in network_list if f in X_scaled.columns]
X_network = X_scaled[available_net]

# Statistical features
def per_row_stat_features(df):
    X_np = df.values
    stats_list = []
    for row in X_np:
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
        stats_list.append([mean_v, median_v, std_v, var_v, range_v, iqr_v,
                           skew_v, kurt_v, entropy_v, zero_ratio, cv_v,
                           q_vals[0], q_vals[1], q_vals[2], q_vals[3], q_vals[4], q_vals[5]])
    cols = ['mean','median','std','variance','range','iqr','skewness','kurtosis',
            'entropy','zero_ratio','cv','q10','q25','q50','q75','q90','q95']
    return pd.DataFrame(stats_list, columns=cols, index=df.index)

X_stat = per_row_stat_features(X_scaled)
X_combined = pd.concat([X_network, X_stat], axis=1)
print(f"Combined features shape: {X_combined.shape}")

# Labels
le = LabelEncoder()
y = le.fit_transform(df_raw['Label'])
print(f"Label classes: {le.classes_}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42, stratify=y)

# ================================
# 2. TRAIN MODELS
# ================================
print("\n" + "="*80)
print("STEP 2: Training models")
print("="*80)

# Random Forest
rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
rf.fit(X_train, y_train)
print("Random Forest trained.")

# XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=50, random_state=42, eval_metric='mlogloss', n_jobs=1)
xgb_model.fit(X_train, y_train)
print("XGBoost trained.")

# Isolation Forest (trained on benign only)
benign_train = X_train[y_train == 0]
iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=1)
iso.fit(benign_train)
print("Isolation Forest trained on benign samples.")

# ================================
# 3. GENERATE SYNTHETIC ANOMALIES
# ================================
print("\n" + "="*80)
print("STEP 3: Generating synthetic zero-day anomalies")
print("="*80)

attack_samples = X_test[y_test != 0]
n_anomalies = 500
noise_scale = 0.05
anomalies = []
for _ in range(n_anomalies):
    idx = np.random.choice(len(attack_samples))
    base = attack_samples.iloc[idx].values
    noise = np.random.normal(0, noise_scale, size=base.shape)
    anomaly = base + noise
    anomaly = np.clip(anomaly, 0, 1)
    anomalies.append(anomaly)
anomalies = np.array(anomalies)
print(f"Generated {n_anomalies} anomalies.")

# Augment test set
X_augmented = np.vstack([X_test, anomalies])
y_augmented = np.hstack([y_test, np.full(n_anomalies, 999)])  # 999 = anomaly label

# ================================
# 4. ISOLATION FOREST THRESHOLD
# ================================
benign_scores = iso.decision_function(benign_train)
threshold = np.percentile(benign_scores, 5)   # 5th percentile
iso_scores = iso.decision_function(X_augmented)
iso_anomaly_flag = (iso_scores < threshold).astype(int)   # 1 = anomaly

# ================================
# 5. EVALUATE MODELS
# ================================
print("\n" + "="*80)
print("STEP 4: Evaluation on real test data + anomalies")
print("="*80)

results = {}
anomaly_mask = (y_augmented == 999)
real_mask = ~anomaly_mask

# Helper to compute metrics for a given model's predictions
def evaluate(y_true, y_pred, model_name):
    # On real data (no anomalies)
    real_true = y_true[real_mask]
    real_pred = y_pred[real_mask]
    acc = accuracy_score(real_true, real_pred)
    prec = precision_score(real_true, real_pred, average='weighted', zero_division=0)
    rec = recall_score(real_true, real_pred, average='weighted')
    f1 = f1_score(real_true, real_pred, average='weighted')
    # Per-class F1 (excluding anomaly class 999)
    report = classification_report(real_true, real_pred, output_dict=True, zero_division=0)
    
    # Anomaly detection (for anomalies only)
    anomaly_true = np.ones(n_anomalies)  # all anomalies are positive
    anomaly_pred = y_pred[anomaly_mask]
    anomaly_detected = (anomaly_pred != 0).sum()  # count where predicted as attack (non-benign)
    anomaly_recall = anomaly_detected / n_anomalies
    
    # Confusion matrix for anomalies
    cm_anomaly = confusion_matrix(anomaly_true, (anomaly_pred != 0).astype(int))
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'per_class_f1': {cls: report[str(i)]['f1-score'] for i, cls in enumerate(le.classes_) if str(i) in report},
        'anomaly_recall': anomaly_recall,
        'anomaly_cm': cm_anomaly
    }

# ---- Random Forest ----
rf_pred = rf.predict(X_augmented)
results['Random Forest'] = evaluate(y_augmented, rf_pred, 'RF')

# ---- XGBoost ----
xgb_pred = xgb_model.predict(X_augmented)
results['XGBoost'] = evaluate(y_augmented, xgb_pred, 'XGB')

# ---- Hybrid (Intelligent Override) ----
hybrid_pred = rf_pred.copy()
for i in range(len(X_augmented)):
    if iso_anomaly_flag[i] == 1:
        # If RF already predicted non-benign (class != 0), keep it.
        # If RF predicted benign, override to the most probable attack class.
        if hybrid_pred[i] == 0:
            proba = rf.predict_proba(X_augmented[i].reshape(1,-1))[0]
            # Find class with highest probability among attack classes (indices 1..4)
            attack_probs = proba[1:]
            if attack_probs.max() > 0:
                best_attack_class = np.argmax(attack_probs) + 1
            else:
                best_attack_class = 1   # fallback
            hybrid_pred[i] = best_attack_class
        # else: keep as is (already attack)
results['Hybrid (IF+RF)'] = evaluate(y_augmented, hybrid_pred, 'Hybrid')

# Print results
print("\n--- Performance on Real Test Data (no anomalies) ---")
for name, res in results.items():
    print(f"\n{name}:")
    # Only accuracy is rounded to 2 decimal places
    print(f"  Accuracy: {res['accuracy']:.2f}")
    print(f"  Weighted Precision: {res['precision']:.4f}")
    print(f"  Weighted Recall: {res['recall']:.4f}")
    print(f"  Weighted F1: {res['f1']:.4f}")
    print(f"  Per-class F1: {res['per_class_f1']}")

print("\n--- Anomaly Detection on Synthetic Zero-Day Attacks ---")
for name, res in results.items():
    print(f"{name}: Recall = {res['anomaly_recall']:.4f} ({int(res['anomaly_recall']*n_anomalies)}/{n_anomalies})")
    print(f"  Confusion matrix (anomaly vs predicted non-benign):\n{res['anomaly_cm']}")

# ================================
# 6. VISUALIZATIONS
# ================================
print("\n" + "="*80)
print("STEP 5: Generating visualizations")
print("="*80)

# Bar chart: Anomaly recall
models = list(results.keys())
anomaly_recalls = [results[m]['anomaly_recall'] for m in models]
plt.figure(figsize=(8,5))
bars = plt.bar(models, anomaly_recalls, color=['red', 'orange', 'green'])
plt.ylabel('Anomaly Detection Recall')
plt.title('Detection of Synthetic Zero-Day Anomalies')
plt.ylim(0, 1.05)
for bar, val in zip(bars, anomaly_recalls):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}', ha='center')
plt.savefig('anomaly_recall_all_models.png', dpi=150)
plt.close()
print("Saved: anomaly_recall_all_models.png")

# Bar chart: Accuracy on real data (show 2 decimal places on bars)
real_acc = [results[m]['accuracy'] for m in models]
plt.figure(figsize=(8,5))
bars = plt.bar(models, real_acc, color=['red', 'orange', 'green'])
plt.ylabel('Accuracy on Real Test Data')
plt.title('Model Accuracy (Known Attacks)')
plt.ylim(0.7, 0.85)
for bar, val in zip(bars, real_acc):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.005, f'{val:.2f}', ha='center')
plt.savefig('real_accuracy_comparison.png', dpi=150)
plt.close()
print("Saved: real_accuracy_comparison.png")

# ROC curve for Isolation Forest (anomaly detection)
y_true_anomaly = np.where(y_augmented == 999, 1, 0)
fpr, tpr, _ = roc_curve(y_true_anomaly, -iso_scores)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Isolation Forest (AUC = {roc_auc:.3f})')
plt.plot([0,1],[0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Anomaly Detection')
plt.legend()
plt.savefig('iso_forest_roc_final.png', dpi=150)
plt.close()
print("Saved: iso_forest_roc_final.png")

print("\n" + "="*80)
print("Refined demonstration complete. Hybrid uses intelligent override, not forced class 1.")
print("="*80)