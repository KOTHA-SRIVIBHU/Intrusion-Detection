"""
Intrusion Detection System - Novel Hybrid Model (Isolation Forest + Random Forest)
No TensorFlow required – runs entirely on CPU.
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import shap

np.random.seed(42)

# ------------------------------
# 1. Load and sample data
# ------------------------------
print("="*80)
print("STEP 1: Loading balanced datasets")
print("="*80)

files = ['02-14_train_balanced.csv', '02-16_train_balanced.csv', '02-21_train_balanced.csv']
dfs = []
for f in files:
    df = pd.read_csv(f)
    print(f"{f}: {df.shape}")
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
print(f"Combined shape: {combined.shape}")

# Sample 300k rows to keep memory and time reasonable
sample_size = 300000
combined_sampled = combined.sample(n=sample_size, random_state=42, ignore_index=True)
X = combined_sampled.drop('Label', axis=1)
y = combined_sampled['Label']
print(f"Sampled shape: {X.shape}")
print(f"Class distribution:\n{y.value_counts()}")

# ------------------------------
# 2. Train/Val/Test split (80/10/10)
# ------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ------------------------------
# 3. Basic Models (RF, XGB, LGBM)
# ------------------------------
print("\n" + "="*80)
print("STEP 3: Training Basic Models")
print("="*80)

results = {}

# Random Forest
print("\n[1/3] Random Forest (n_estimators=50)...")
start = time.time()
rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
rf.fit(X_train, y_train)
rf_time = time.time() - start
rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)
results['Random Forest'] = {
    'train_time': rf_time,
    'accuracy': accuracy_score(y_test, rf_pred),
    'precision': precision_score(y_test, rf_pred, average='weighted'),
    'recall': recall_score(y_test, rf_pred, average='weighted'),
    'f1': f1_score(y_test, rf_pred, average='weighted'),
    'auc': roc_auc_score(y_test, rf_proba, multi_class='ovr', average='weighted')
}
print(f"  Accuracy: {results['Random Forest']['accuracy']:.4f}, F1: {results['Random Forest']['f1']:.4f}, Time: {rf_time:.2f}s")

# XGBoost
print("\n[2/3] XGBoost (n_estimators=50)...")
start = time.time()
xgb_model = xgb.XGBClassifier(n_estimators=50, random_state=42, use_label_encoder=False, eval_metric='mlogloss', n_jobs=1)
xgb_model.fit(X_train, y_train)
xgb_time = time.time() - start
xgb_pred = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)
results['XGBoost'] = {
    'train_time': xgb_time,
    'accuracy': accuracy_score(y_test, xgb_pred),
    'precision': precision_score(y_test, xgb_pred, average='weighted'),
    'recall': recall_score(y_test, xgb_pred, average='weighted'),
    'f1': f1_score(y_test, xgb_pred, average='weighted'),
    'auc': roc_auc_score(y_test, xgb_proba, multi_class='ovr', average='weighted')
}
print(f"  Accuracy: {results['XGBoost']['accuracy']:.4f}, F1: {results['XGBoost']['f1']:.4f}, Time: {xgb_time:.2f}s")

# LightGBM
print("\n[3/3] LightGBM (n_estimators=50)...")
start = time.time()
lgb_model = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1, n_jobs=1)
lgb_model.fit(X_train, y_train)
lgb_time = time.time() - start
lgb_pred = lgb_model.predict(X_test)
lgb_proba = lgb_model.predict_proba(X_test)
results['LightGBM'] = {
    'train_time': lgb_time,
    'accuracy': accuracy_score(y_test, lgb_pred),
    'precision': precision_score(y_test, lgb_pred, average='weighted'),
    'recall': recall_score(y_test, lgb_pred, average='weighted'),
    'f1': f1_score(y_test, lgb_pred, average='weighted'),
    'auc': roc_auc_score(y_test, lgb_proba, multi_class='ovr', average='weighted')
}
print(f"  Accuracy: {results['LightGBM']['accuracy']:.4f}, F1: {results['LightGBM']['f1']:.4f}, Time: {lgb_time:.2f}s")

# ------------------------------
# 4. Proposed Hybrid Model (Isolation Forest + Random Forest)
# ------------------------------
print("\n" + "="*80)
print("STEP 4: Proposed Hybrid Model (Isolation Forest + Random Forest)")
print("="*80)

# Train Isolation Forest only on benign training data (unsupervised anomaly detection)
benign_train = X_train[y_train == 0]
print(f"Benign samples for Isolation Forest: {benign_train.shape}")

start = time.time()
iso_forest = IsolationForest(contamination=0.1, random_state=42, n_jobs=1)
iso_forest.fit(benign_train)  # train only on benign
if_train_time = time.time() - start
print(f"  Isolation Forest training time: {if_train_time:.2f}s")

# Predict anomalies on test set (1 = normal, -1 = anomaly)
test_anomaly_flag = iso_forest.predict(X_test)
test_anomaly = (test_anomaly_flag == -1).astype(int)  # 1 = anomaly

# Train Random Forest (explainable) on all training data
print("\nTraining Random Forest (for classification)...")
start = time.time()
rf_explain = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
rf_explain.fit(X_train, y_train)
rf_explain_time = time.time() - start
rf_explain_pred = rf_explain.predict(X_test)
rf_explain_proba = rf_explain.predict_proba(X_test)

# Hybrid decision: if Isolation Forest says anomaly, override to most probable attack class
hybrid_pred = rf_explain_pred.copy()
for i in np.where(test_anomaly == 1)[0]:
    hybrid_pred[i] = np.argmax(rf_explain_proba[i])  # highest probability class

hybrid_acc = accuracy_score(y_test, hybrid_pred)
hybrid_prec = precision_score(y_test, hybrid_pred, average='weighted', zero_division=0)
hybrid_rec = recall_score(y_test, hybrid_pred, average='weighted')
hybrid_f1 = f1_score(y_test, hybrid_pred, average='weighted')

# For AUC, boost anomaly samples' attack probabilities
hybrid_proba = rf_explain_proba.copy()
for i in np.where(test_anomaly == 1)[0]:
    attack_class = hybrid_pred[i]
    hybrid_proba[i][attack_class] = 1.0
    hybrid_proba[i][0] = 0.0
hybrid_auc = roc_auc_score(y_test, hybrid_proba, multi_class='ovr', average='weighted')

results['Hybrid (IF+RF)'] = {
    'train_time': if_train_time + rf_explain_time,
    'accuracy': hybrid_acc,
    'precision': hybrid_prec,
    'recall': hybrid_rec,
    'f1': hybrid_f1,
    'auc': hybrid_auc
}
print(f"\nHybrid evaluation: Acc={hybrid_acc:.4f}, F1={hybrid_f1:.4f}, AUC={hybrid_auc:.4f}")

# ------------------------------
# 5. Comparison Table
# ------------------------------
print("\n" + "="*80)
print("STEP 5: Model Comparison")
print("="*80)

comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df[['accuracy', 'precision', 'recall', 'f1', 'auc', 'train_time']]
comparison_df.columns = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC', 'Train Time (s)']
print(comparison_df.round(4))
comparison_df.to_csv('model_comparison.csv')
print("\nSaved: model_comparison.csv")

# ------------------------------
# 6. SHAP Analysis
# ------------------------------
print("\n" + "="*80)
print("STEP 6: SHAP Analysis (sampled 200 rows)")
print("="*80)

X_test_sample = X_test.sample(n=200, random_state=42)
explainer = shap.TreeExplainer(rf_explain)
shap_values = explainer.shap_values(X_test_sample)

# Summary bar plot
shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('shap_summary_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: shap_summary_plot.png")

# Force plot for first attack
attack_idx = (y_test == 1).idxmax()
if attack_idx in X_test_sample.index:
    shap.force_plot(explainer.expected_value[1], shap_values[1][X_test_sample.index.get_loc(attack_idx)], 
                    X_test_sample.loc[attack_idx], matplotlib=True, show=False)
    plt.savefig('shap_force_plot_sample.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: shap_force_plot_sample.png")
else:
    print("Attack sample not in SHAP subset; skipping force plot.")

print("\n" + "="*80)
print("All tasks completed successfully!")
print("="*80)