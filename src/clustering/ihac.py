import logging
from pathlib import Path
from typing import List, Sequence, Set, Tuple, Optional, Dict

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, is_valid_linkage

from clustering.hierarchical_clustering import BaseClusteringWrapper


class iHAC(BaseClusteringWrapper):
    """
    Incremental HAC with MLB constraints.

    This implementation reduces runtime by:
    - compressing unconstrained duplicate vectors at initialization
    - checking only constraint subsets relevant to each merge candidate
    - plotting via compressed active-cluster representatives
    """

    def __init__(
        self,
        X: np.ndarray,
        constraints: Optional[Sequence[Tuple[int, int, int]]] = None,
        *,
        figsize: tuple[int, int] = (10, 6),
    ) -> None:
        """
        Args:
            X: n x d feature matrix.
            constraints: MLB triples (i, j, k), i.e. i and j should merge before k.
            figsize: Matplotlib figure size for plotting.
        """
        super().__init__(figsize=figsize)
        self.X = X
        self.n = X.shape[0]
        self.constraints = [tuple(map(int, c)) for c in constraints] if constraints else []

        # Cluster bookkeeping
        self.clusters: Dict[int, Set[int]] = {}
        self.active_clusters: Set[int] = set()
        self.point_to_cluster: Dict[int, int] = {}
        self.next_cluster_id = self.n

        # Track constraint violations that already occurred
        self.violated_constraints = set()

        # Cluster stats for similarity
        self.cluster_sizes: Dict[int, int] = {}
        self.cluster_centroids: Dict[int, np.ndarray] = {}

        # Track merges for dendrogram plotting
        self.merge_history = []
        self.snapshots = []
        self.reduced_snapshots = []
        self.initial_representative_points: List[int] = []
        self.initial_cluster_ids: List[int] = []
        self.cluster_rep_members: Dict[int, Set[int]] = {}
        self.cluster_rep_sizes: Dict[int, int] = {}

        self._initialize_clusters()

        # Map active cluster -> constraints involving at least one member point.
        # This lets violation checks inspect only relevant constraints for a candidate merge.
        self.cluster_to_constraint_idxs: Dict[int, Set[int]] = {
            cid: set() for cid in self.active_clusters
        }
        for idx, (a, b, c) in enumerate(self.constraints):
            for point_idx in (a, b, c):
                if point_idx < 0 or point_idx >= self.n:
                    raise ValueError(
                        f"Constraint point index {point_idx} out of bounds for n={self.n}."
                    )
                cid = self.point_to_cluster[point_idx]
                self.cluster_to_constraint_idxs[cid].add(idx)

    def _add_initial_cluster(self, cluster_id: int, members: Set[int]) -> None:
        """Add one initial active cluster and register its representative index."""
        self.clusters[cluster_id] = members
        self.active_clusters.add(cluster_id)
        for point_idx in members:
            self.point_to_cluster[point_idx] = cluster_id
        self.cluster_sizes[cluster_id] = len(members)
        # Members are initialized from identical rows, so one representative centroid is enough.
        representative_idx = next(iter(members))
        self.cluster_centroids[cluster_id] = self.X[representative_idx].copy()
        self.initial_cluster_ids.append(cluster_id)
        rep_idx = len(self.initial_representative_points)
        self.initial_representative_points.append(representative_idx)
        self.cluster_rep_members[cluster_id] = {rep_idx}
        self.cluster_rep_sizes[cluster_id] = 1

    def _initialize_clusters(self) -> None:
        """
        Build initial clusters with duplicate compression.

        Optimization:
        - unconstrained points with identical feature vectors are merged immediately
          into one active cluster.
        - constrained points are kept as singleton seeds to preserve MLB semantics.
        """
        constrained_points = {
            p for triple in self.constraints for p in triple if 0 <= p < self.n
        }
        groups: Dict[bytes, List[int]] = {}
        for idx in range(self.n):
            key = self.X[idx].tobytes()
            groups.setdefault(key, []).append(idx)

        for members in groups.values():
            constrained = [idx for idx in members if idx in constrained_points]
            unconstrained = [idx for idx in members if idx not in constrained_points]

            for point_idx in constrained:
                self._add_initial_cluster(point_idx, {point_idx})

            if unconstrained:
                self._add_initial_cluster(unconstrained[0], set(unconstrained))

        logging.info(
            "iHAC initialization: %d points -> %d active clusters (%d constraints).",
            self.n,
            len(self.active_clusters),
            len(self.constraints),
        )

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

    def _mapped_cluster_id(
        self, cluster_id: int, merge_a: int, merge_b: int, new_id: int
    ) -> int:
        """
        Map old cluster IDs to the new cluster for violation checks.
        If cluster_id is part of merge, return new_id; else return cluster_id.

        Args:
            cluster_id: Cluster ID to map.
            merge_a: First cluster being merged.
            merge_b: Second cluster being merged.
            new_id: New cluster ID.
        """
        # replace merged clusters with cluster ID on highest dendogram level
        if cluster_id == merge_a or cluster_id == merge_b:
            return new_id
        return cluster_id

    def _new_violations_count(self, merge_a: int, merge_b: int) -> int:
        """
        Count newly violated constraints if merge_a and merge_b merge.

        Args:
            merge_a: First cluster being merged.
            merge_b: Second cluster being merged.

        Notes:
            Only constraints touching merge_a or merge_b are evaluated.
            This avoids scanning all constraints for every candidate pair.
        """
        new_id = -1
        count = 0
        relevant_constraints = self.cluster_to_constraint_idxs.get(
            merge_a, set()
        ) | self.cluster_to_constraint_idxs.get(merge_b, set())
        for idx in relevant_constraints:
            a, b, c = self.constraints[idx]
            # skip already violated constraints
            if idx in self.violated_constraints:
                continue
            # get current cluster assignments
            ca = self.point_to_cluster[a]
            cb = self.point_to_cluster[b]
            cc = self.point_to_cluster[c]
            # already clustered -> no violation possible
            if ca == cb:
                continue
            # returns -1 if cluster ca/cb/cc was merged, else returns current cluster id (-> detector of membership after merge)
            ca_m = self._mapped_cluster_id(ca, merge_a, merge_b, new_id)
            cb_m = self._mapped_cluster_id(cb, merge_a, merge_b, new_id)
            cc_m = self._mapped_cluster_id(cc, merge_a, merge_b, new_id)
            # check if ca or cb was merged with cc -> if yes, violates the constraint
            if ca_m != cb_m and (cc_m == ca_m or cc_m == cb_m):
                count += 1
        return count

    def _update_violated_constraints(self) -> None:
        """Mark constraints violated by the current partition."""
        for idx, (a, b, c) in enumerate(self.constraints):
            # no need to double-check already violated constraints
            if idx in self.violated_constraints:
                continue
            # obtain current cluster assignments
            ca = self.point_to_cluster[a]
            cb = self.point_to_cluster[b]
            # already clustered -> no violation possible
            if ca == cb:
                continue
            # cluster assignment of point c
            cc = self.point_to_cluster[c]
            # check if c is now in the same cluster as a or b -> mark as violated
            if cc == ca or cc == cb:
                self.violated_constraints.add(idx)

    def merge(self, i: int, j: int, merge_sim: Optional[float] = None) -> None:
        """
        Merge two active clusters and update internal bookkeeping.

        Args:
            i: First cluster ID.
            j: Second cluster ID.
            merge_sim: Similarity score of the chosen merge pair.
        """
        # add new entry at end of each existing row, add empty row at bottom (upper triangular matrix)
        new_id = self.next_cluster_id
        self.next_cluster_id += 1
        # set of data point indices in the new cluster
        new_cluster = self.clusters[i] | self.clusters[j]
        self.clusters[new_id] = new_cluster
        rep_size_i = self.cluster_rep_sizes[i]
        rep_size_j = self.cluster_rep_sizes[j]
        new_rep_size = rep_size_i + rep_size_j
        merge_distance = 0.0 if merge_sim is None else float(max(0.0, -merge_sim))
        # (left_id, right_id, new_id, distance, representative_count)
        self.merge_history.append((i, j, new_id, merge_distance, new_rep_size))
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

        # Constraints touched by either merged cluster now belong to the merged cluster.
        touched_constraints = self.cluster_to_constraint_idxs.pop(i, set()) | self.cluster_to_constraint_idxs.pop(j, set())
        self.cluster_to_constraint_idxs[new_id] = touched_constraints
        self.cluster_rep_members[new_id] = (
            self.cluster_rep_members.pop(i, set()) | self.cluster_rep_members.pop(j, set())
        )
        self.cluster_rep_sizes[new_id] = new_rep_size

        # Retire old clusters
        for old_id in (i, j):
            self.active_clusters.remove(old_id)
            self.clusters.pop(old_id, None)
            self.cluster_sizes.pop(old_id, None)
            self.cluster_centroids.pop(old_id, None)
            self.cluster_rep_sizes.pop(old_id, None)

        self.active_clusters.add(new_id)
        self._update_violated_constraints()
        self.snapshots.append(
            [set(self.clusters[cid]) for cid in sorted(self.active_clusters)]
        )
        self.reduced_snapshots.append(
            [set(self.cluster_rep_members[cid]) for cid in sorted(self.active_clusters)]
        )

    def run(self) -> List[Set[int]]:
        """
        Run iHAC until a single cluster remains.

        Returns:
            Final partition as a list of clusters.
        """
        # active clusters contain IDs of clusters not yet merged
        while len(self.active_clusters) > 1:
            # initialize tracking of best pair to merge: initially infinite violations; merging any pair is better
            min_viol = float("inf")
            best_pair = None
            best_sim = -float("inf")
            active_list = sorted(self.active_clusters)
            logging.info("Number of active clusters: %d", len(active_list))
            # try all pairs of active clusters
            for idx_i, i in enumerate(active_list):
                for j in active_list[idx_i + 1 :]:
                    # number of new violations if i and j are merged
                    violations = self._new_violations_count(i, j)
                    sim = self.sim(i, j)
                    # track best pair (least violations, then highest similarity)
                    if violations < min_viol or (
                        violations == min_viol and sim > best_sim
                    ):
                        min_viol = violations
                        best_sim = sim
                        best_pair = (i, j)
            if best_pair is None:
                logging.warning("No valid merge found; stopping early.")
                break
            i, j = best_pair
            self.merge(i, j, merge_sim=best_sim)
        return [self.clusters[cid] for cid in sorted(self.active_clusters)]

    def clustering_steps(self) -> List[List[Set[int]]]:
        """
        Return clustering states over original points after each merge.

        Returns:
            List of partitions after each merge.
        """
        return list(self.snapshots)

    def representative_data(self) -> np.ndarray:
        """Return one representative row per initial active cluster."""
        if not self.initial_representative_points:
            return np.empty((0, self.X.shape[1]))
        return self.X[np.array(self.initial_representative_points, dtype=int)]

    def active_cluster_steps(self, include_initial: bool = True) -> List[List[Set[int]]]:
        """
        Return partitions over representative indices (compressed view).

        This is designed for lightweight plotting after duplicate compression.
        If `include_initial` is True, step 1 is the initial active-cluster state.
        """
        steps = list(self.reduced_snapshots)
        if include_initial:
            initial = [{rep_idx} for rep_idx in range(len(self.initial_representative_points))]
            return [initial] + steps
        return steps

    def save_step_dendrogram(
        self,
        out_path: Path,
        *,
        dataset_name: str,
        method_name: str,
        filename: str = "dendrogram_steps.svg",
    ) -> Path:
        """
        Save a styled dendrogram based on the actual iHAC merge sequence.

        Leaves represent compressed initial active clusters.
        Heights use merge-step indices so all tree levels stay visible even when
        constrained distances collapse to similar values.
        """
        out_path.mkdir(parents=True, exist_ok=True)
        n_leaves = len(self.initial_cluster_ids)
        if n_leaves <= 1:
            raise ValueError("Need at least two initial active clusters for a dendrogram.")
        if len(self.merge_history) != n_leaves - 1:
            raise ValueError(
                "Incomplete merge history. Run iHAC to completion before saving dendrogram."
            )
        cluster_to_dendro_id = {
            cid: idx for idx, cid in enumerate(self.initial_cluster_ids)
        }
        linkage_rows = []
        for row_idx, (left, right, new_id, dist, rep_count) in enumerate(self.merge_history):
            linkage_rows.append(
                [cluster_to_dendro_id[left], cluster_to_dendro_id[right], dist, rep_count]
            )
            cluster_to_dendro_id[new_id] = n_leaves + row_idx
        linkage_matrix = np.asarray(linkage_rows, dtype=float)
        # Use merge step as linkage height to visualize the complete iHAC tree.
        # Distances in constrained iHAC can collapse to a single value, which hides levels.
        linkage_matrix[:, 2] = np.arange(1, linkage_matrix.shape[0] + 1, dtype=float)
        if not is_valid_linkage(linkage_matrix, throw=False, warning=False):
            raise ValueError("Invalid linkage matrix built from iHAC merge history.")

        fig = plt.figure(figsize=(max(self.figsize[0], 12), max(self.figsize[1], 6)))
        ax = fig.add_subplot(111)
        color_threshold = 0.7 * np.max(linkage_matrix[:, 2]) if linkage_matrix.size else 0
        dendrogram(
            linkage_matrix,
            ax=ax,
            labels=[f"C{idx}" for idx in range(n_leaves)],
            leaf_rotation=90,
            leaf_font_size=8,
            color_threshold=color_threshold,
            above_threshold_color="#6b6b6b",
        )
        ax.set_xlabel("Compressed Active Clusters")
        ax.set_ylabel("Merge Step")
        ax.set_title(f"{dataset_name} - {method_name} - iHAC Step Dendrogram")
        if linkage_matrix.shape[0] > 0 and color_threshold > 0:
            ax.axhline(
                y=color_threshold,
                color="#8a8a8a",
                linestyle="--",
                linewidth=0.9,
            )
            ax.text(
                0.01,
                0.98,
                f"color threshold: {color_threshold:.1f}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )
        plt.tight_layout()
        svg_path = out_path / f"{dataset_name}_{method_name}_{filename}"
        plt.savefig(svg_path, format="svg")
        plt.close()
        logging.info("Saved iHAC step dendrogram: %s", svg_path)
        return svg_path

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
