from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris

def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)

    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    dendrogram(linkage_matrix, **kwargs)

# Load Iris data
iris = load_iris()
X = iris.data
y = iris.target

# Colors for clusters/species
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:cyan"]

# Create figure with multiple subplots
min_n = 3
max_n = 5
n_subplots = 2 + (max_n - min_n)
fig, axes = plt.subplots(1, n_subplots, figsize=(10*n_subplots, 6))

# ---- Scatter plot with ground truth IDs ----
axes[0].scatter(X[:, 0], X[:, 1], c=[colors[i] for i in y])

for i in range(len(X)):
    axes[0].annotate(
        str(i),
        (X[i, 0], X[i, 1]),
        fontsize=6,
        alpha=0.7,
        xytext=(2, 2),
        textcoords="offset points"
    )

# Add legend for ground truth
handles = [Patch(color=colors[i], label=f"Class {i}") for i in np.unique(y)]
axes[0].legend(handles=handles)

axes[0].set_xlabel("Sepal length")
axes[0].set_ylabel("Sepal width")
axes[0].set_title("Iris Scatter Plot with Sample IDs (Ground Truth)")

# ---- Dendrogram ----
model_full = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model_full.fit(X)

plt.sca(axes[1])
plot_dendrogram(model_full, truncate_mode="level", p=3)
axes[1].set_title("Hierarchical Clustering Dendrogram")
axes[1].set_xlabel(
    "Number of points in node (or index of point if no parenthesis)"
)

# ---- Scatter plots of clusters for n = 3..max_n ----
cache_path = Path("tmp/cache")
cache_path.mkdir(parents=True, exist_ok=True)

for n in range(min_n, max_n):
    model_i = AgglomerativeClustering(n_clusters=n, memory=str(cache_path), compute_full_tree=True)
    cluster_labels = model_i.fit_predict(X)
    pos = 2 + (n - min_n)

    axes[pos].scatter(X[:, 0], X[:, 1], c=[colors[i] for i in cluster_labels])

    for i in range(len(X)):
        axes[pos].annotate(
            str(i),
            (X[i, 0], X[i, 1]),
            fontsize=6,
            alpha=0.7,
            xytext=(2, 2),
            textcoords="offset points"
        )

    # Add legend for this cluster scatter
    handles = [Patch(color=colors[i], label=f"Class {i}") for i in range(n)]
    axes[pos].legend(handles=handles)

    axes[pos].set_xlabel("Sepal length")
    axes[pos].set_ylabel("Sepal width")
    axes[pos].set_title(f"Iris Scatter Plot with {n}-Cluster Labels")

plt.tight_layout()
plt.show()
