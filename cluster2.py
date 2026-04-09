"""
Clustering Analysis for Intrusion Detection
- Loads combined features (network + statistical) from CSV
- Applies K-means clustering
- Visualizes clusters using PCA and t-SNE
- Evaluates cluster purity against true labels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# 1. Load combined features and labels
# ------------------------------
print("="*80)
print("Loading combined features and labels...")
print("="*80)

# Load features (36 columns)
X = pd.read_csv('combined_features.csv')
print(f"Features shape: {X.shape}")

# We need the original labels for comparison. They are in the same order as rows in combined_features.csv.
# Since we saved combined_features.csv from the original df_raw, we can reload the raw data to get labels.
# Alternatively, we saved y_all earlier. For simplicity, reload a small portion of original data to get labels.
# But easier: during preprocessing we had y_all. To avoid re-running, we'll load from the saved combined_features.csv
# and also load the original raw combined data to extract labels (same order). 
# Actually, combined_features.csv was created from X_combined which had the same index as y_all.
# We'll just reload the first 60k rows of original CSVs to get labels (same order as we concatenated).

print("Reloading original labels (from first 20k each)...")
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
y = df_raw['Label']
print(f"Labels shape: {len(y)}")
print(f"Class distribution:\n{y.value_counts()}")

# Ensure same length
assert len(X) == len(y), "Mismatch between features and labels"

# ------------------------------
# 2. K-means clustering
# ------------------------------
print("\n" + "="*80)
print("Applying K-means clustering...")
print("="*80)

# Determine optimal number of clusters using elbow method
inertias = []
K_range = range(2, 10)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.savefig('elbow_plot.png', dpi=150)
plt.close()
print("Saved: elbow_plot.png")

# Choose k=5 (since we have 5 attack classes? Actually we have 5 distinct labels including Benign)
k_optimal = 5
print(f"Using k = {k_optimal} clusters")

kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

# ------------------------------
# 3. Evaluation against true labels
# ------------------------------
print("\n" + "="*80)
print("Clustering Evaluation")
print("="*80)

# Map cluster labels to true labels (best alignment)
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score

ari = adjusted_rand_score(y, cluster_labels)
nmi = normalized_mutual_info_score(y, cluster_labels)
homogeneity = homogeneity_score(y, cluster_labels)
silhouette = silhouette_score(X, cluster_labels)

print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
print(f"Homogeneity: {homogeneity:.4f}")
print(f"Silhouette Score: {silhouette:.4f}")

# Cross-tabulation
cross_tab = pd.crosstab(y, cluster_labels)
print("\nConfusion between true labels and clusters:")
print(cross_tab)

# Save cross-tab
cross_tab.to_csv('cluster_label_confusion.csv')
print("\nSaved: cluster_label_confusion.csv")

# ------------------------------
# 4. Visualization using PCA and t-SNE
# ------------------------------
print("\n" + "="*80)
print("Generating visualizations...")
print("="*80)

# PCA to 2D
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=cluster_labels, cmap='viridis', alpha=0.6, s=10)
plt.colorbar(scatter, label='Cluster ID')
plt.title('K-means Clusters (PCA projection)')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.subplot(1,2,2)
# Map labels to numeric for coloring
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_numeric = le.fit_transform(y)
scatter2 = plt.scatter(X_pca[:,0], X_pca[:,1], c=y_numeric, cmap='tab10', alpha=0.6, s=10)
plt.colorbar(scatter2, label='True Attack Class')
plt.title('True Labels (PCA projection)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.savefig('pca_clusters_vs_true.png', dpi=150)
plt.close()
print("Saved: pca_clusters_vs_true.png")

# t-SNE (more accurate but slower; sample 10k rows for speed)
print("Computing t-SNE on 10,000 random samples...")
sample_idx = np.random.choice(len(X), size=10000, replace=False)
X_sample = X.iloc[sample_idx]
y_sample = y.iloc[sample_idx]
cluster_sample = cluster_labels[sample_idx]

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_sample)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=cluster_sample, cmap='viridis', alpha=0.6, s=10)
plt.colorbar(label='Cluster ID')
plt.title('K-means Clusters (t-SNE)')

plt.subplot(1,2,2)
y_sample_numeric = le.transform(y_sample)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y_sample_numeric, cmap='tab10', alpha=0.6, s=10)
plt.colorbar(label='True Attack Class')
plt.title('True Labels (t-SNE)')
plt.tight_layout()
plt.savefig('tsne_clusters_vs_true.png', dpi=150)
plt.close()
print("Saved: tsne_clusters_vs_true.png")

print("\n" + "="*80)
print("Clustering analysis completed. Check generated plots and CSV.")
print("="*80)
