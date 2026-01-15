import numpy as np
import matplotlib.pyplot as plt


class iHAC:
    def __init__(self, X, constraints=None):
        """
        X: n x d feature matrix
        constraints: list of tuples (i, j, k)
                     meaning i & j must merge before k
        """
        self.X = X
        self.n = X.shape[0]
        self.constraints = constraints if constraints else []

        # Active clusters and point membership
        self.clusters = {i: {i} for i in range(self.n)}
        self.active_clusters = set(range(self.n))
        self.point_to_cluster = {i: i for i in range(self.n)}

        # Track constraint violations that already occurred
        self.violated_constraints = set()

        # Cluster stats for similarity
        self.cluster_sizes = {i: 1 for i in range(self.n)}
        self.cluster_centroids = {i: self.X[i].copy() for i in range(self.n)}

        # Track merges for dendrogram plotting
        self.merge_history = []
        self.snapshots = []

    def sim(self, i, j):
        """UPGMA similarity: negative Euclidean distance"""
        xi = self.cluster_centroids[i]
        xj = self.cluster_centroids[j]
        return -np.linalg.norm(xi - xj)

    def _mapped_cluster_id(self, cluster_id, merge_a, merge_b, new_id):
        if cluster_id == merge_a or cluster_id == merge_b:
            return new_id
        return cluster_id

    def _new_violations_count(self, merge_a, merge_b):
        new_id = -1
        count = 0
        for idx, (a, b, c) in enumerate(self.constraints):
            if idx in self.violated_constraints:
                continue
            ca = self.point_to_cluster[a]
            cb = self.point_to_cluster[b]
            cc = self.point_to_cluster[c]
            if ca == cb:
                continue
            ca_m = self._mapped_cluster_id(ca, merge_a, merge_b, new_id)
            cb_m = self._mapped_cluster_id(cb, merge_a, merge_b, new_id)
            cc_m = self._mapped_cluster_id(cc, merge_a, merge_b, new_id)
            if ca_m != cb_m and (cc_m == ca_m or cc_m == cb_m):
                count += 1
        return count

    def _update_violated_constraints(self):
        for idx, (a, b, c) in enumerate(self.constraints):
            if idx in self.violated_constraints:
                continue
            ca = self.point_to_cluster[a]
            cb = self.point_to_cluster[b]
            if ca == cb:
                continue
            cc = self.point_to_cluster[c]
            if cc == ca or cc == cb:
                self.violated_constraints.add(idx)

    def merge(self, i, j):
        new_id = max(self.clusters.keys()) + 1
        new_cluster = self.clusters[i] | self.clusters[j]
        self.clusters[new_id] = new_cluster
        self.merge_history.append((i, j, new_id))

        # Update cluster stats
        size_i = self.cluster_sizes[i]
        size_j = self.cluster_sizes[j]
        centroid_i = self.cluster_centroids[i]
        centroid_j = self.cluster_centroids[j]
        new_size = size_i + size_j
        new_centroid = (centroid_i * size_i + centroid_j * size_j) / new_size
        self.cluster_sizes[new_id] = new_size
        self.cluster_centroids[new_id] = new_centroid

        # Update point memberships
        for point in new_cluster:
            self.point_to_cluster[point] = new_id

        # Retire old clusters
        for old_id in (i, j):
            self.active_clusters.remove(old_id)
            self.clusters.pop(old_id, None)
            self.cluster_sizes.pop(old_id, None)
            self.cluster_centroids.pop(old_id, None)

        self.active_clusters.add(new_id)
        self._update_violated_constraints()
        self.snapshots.append([set(self.clusters[cid]) for cid in sorted(self.active_clusters)])

    def run(self):
        while len(self.active_clusters) > 1:
            min_viol = float("inf")
            best_pair = None
            best_sim = -float("inf")
            active_list = sorted(self.active_clusters)
            for idx_i, i in enumerate(active_list):
                for j in active_list[idx_i + 1:]:
                    violations = self._new_violations_count(i, j)
                    sim = self.sim(i, j)
                    if violations < min_viol or (violations == min_viol and sim > best_sim):
                        min_viol = violations
                        best_sim = sim
                        best_pair = (i, j)
            if best_pair is None:
                break
            i, j = best_pair
            self.merge(i, j)
        return [self.clusters[cid] for cid in sorted(self.active_clusters)]

    def clustering_steps(self):
        """Return clustering after each merge as a list of partitions."""
        return list(self.snapshots)


# -------------------------------
# Example usage
X = np.array([
    [1.0, 2.0],
    [1.5, 1.8],
    [5.0, 8.0],
    [8.0, 8.0],
    [1.2, 0.9]
])
constraints = [
    (0, 1, 2),
    (0, 4, 3)
]

ihac = iHAC(X, constraints)
clusters = ihac.run()
ihac_steps = ihac.clustering_steps()
print("Clustering steps:", ihac_steps)
print("Clusters:", clusters)

# Map cluster IDs to 0..num_clusters-1 for colors
for clusters in ihac_steps:
    num_clusters = len(clusters)
    cluster_id_map = {cid: i for i, cid in enumerate(range(num_clusters))}
    cluster_labels = np.zeros(X.shape[0], dtype=int)
    for cid, cluster in enumerate(clusters):
        for idx in cluster:
            cluster_labels[idx] = cluster_id_map[cid]

    # Pick enough colors
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
            "tab:brown", "tab:pink"]

    plt.figure(figsize=(7, 5))

    # Plot points
    for i in range(len(X)):
        plt.scatter(X[i, 0], X[i, 1], color=colors[cluster_labels[i]], s=100)
        plt.text(X[i, 0] + 0.05, X[i, 1] + 0.05, f"$x_{{{i}}}$", fontsize=10)

    # Create legend entries with cluster and relevant constraints
    legend_labels = []
    for cid, cluster in enumerate(clusters):
        # Find constraints that involve points in this cluster
        relevant_constraints = []
        for a, b, c in constraints:
            if a in cluster or b in cluster:
                relevant_constraints.append(f"$x_{{{a}}}, x_{{{b}}} \\rightarrow x_{{{c}}}$")
        label = f"$\\mathrm{{Cluster\\ {cid}}}$"
        if relevant_constraints:
            label += ": " + ", ".join(relevant_constraints)
        legend_labels.append(label)
        plt.scatter([], [], color=colors[cid], label=label)  # dummy for legend

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("iHAC Clustering Result with Constraints")
    plt.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()
