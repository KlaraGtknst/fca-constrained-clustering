import logging
from typing import List, Sequence, Set, Tuple, Optional, Dict

import numpy as np

from clustering.hierarchical_clustering import BaseClusteringWrapper


class iHAC(BaseClusteringWrapper):
    """Incremental Hierarchical Agglomerative Clustering with constraints."""

    def __init__(
        self,
        X: np.ndarray,
        constraints: Optional[Sequence[Tuple[int, int, int]]] = None,
        *,
        figsize: tuple[int, int]=(10, 6),
    ) -> None:
        """
        Args:
            X: n x d feature matrix.
            constraints: List of tuples (i, j, k) meaning i & j must merge before k.
            figsize: Matplotlib figure size for plotting.
        """
        super().__init__(figsize=figsize)
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

    def sim(self, i: int, j: int) -> float:
        """
        Return UPGMA similarity as negative Euclidean distance.

        Args:
            i: Cluster ID.
            j: Cluster ID.
        """
        xi = self.cluster_centroids[i]
        xj = self.cluster_centroids[j]
        return -np.linalg.norm(xi - xj)

    def _mapped_cluster_id(self, cluster_id: int, merge_a: int, merge_b: int, new_id: int) -> int:
        """
        Map old cluster IDs to the new cluster for violation checks.

        Args:
            cluster_id: Cluster ID to map.
            merge_a: First cluster being merged.
            merge_b: Second cluster being merged.
            new_id: New cluster ID.
        """
        if cluster_id == merge_a or cluster_id == merge_b:
            return new_id
        return cluster_id

    def _new_violations_count(self, merge_a: int, merge_b: int) -> int:
        """
        Count newly violated constraints if merge_a and merge_b merge.

        Args:
            merge_a: First cluster being merged.
            merge_b: Second cluster being merged.
        """
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

    def _update_violated_constraints(self) -> None:
        """Mark constraints violated by the current partition."""
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

    def merge(self, i: int, j: int) -> None:
        """
        Merge two active clusters and update internal bookkeeping.

        Args:
            i: First cluster ID.
            j: Second cluster ID.
        """
        new_id = max(self.clusters.keys()) + 1
        new_cluster = self.clusters[i] | self.clusters[j]
        self.clusters[new_id] = new_cluster
        self.merge_history.append((i, j, new_id))
        logging.info("Merged clusters %s and %s into %s", i, j, new_id)

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

    def run(self) -> List[Set[int]]:
        """
        Run iHAC until a single cluster remains.

        Returns:
            Final partition as a list of clusters.
        """
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
                logging.warning("No valid merge found; stopping early.")
                break
            i, j = best_pair
            self.merge(i, j)
        return [self.clusters[cid] for cid in sorted(self.active_clusters)]

    def clustering_steps(self) -> List[List[Set[int]]]:
        """
        Return clustering after each merge as a list of partitions.

        Returns:
            List of partitions after each merge.
        """
        return list(self.snapshots)

    def cluster(self, data: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Run iHAC and return labels per step.

        Args:
            data: Input data (must match the data used at initialization).

        Returns:
            Mapping from step index to label array.
        """
        if data is not self.X:
            raise ValueError("iHAC expects the same data passed during initialization.")
        self.run()
        partitions = self.clustering_steps()
        return {
            idx + 1: self._labels_from_partition(partition, self.n)
            for idx, partition in enumerate(partitions)
        }
