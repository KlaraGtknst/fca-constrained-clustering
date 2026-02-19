import logging
import csv
from pathlib import Path
from typing import List, Sequence, Set, Tuple, Optional, Dict

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, is_valid_linkage

from clustering.hierarchical_clustering import BaseClusteringWrapper

ConstraintTriple = Tuple[int, int, int]
EquivConstraintTriple = Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]


class iHAC(BaseClusteringWrapper):
    """
    Incremental HAC with MLB constraints.

    This implementation reduces runtime by:
    - compressing duplicate vectors at initialization
    - checking only constraint subsets relevant to each merge candidate
    - evaluating MLB constraints lazily over row-equivalence classes
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
        Initialize iHAC state and precompute structures needed for fast constrained merges.

        Parameters
        ----------
        X:
          Feature matrix (`n_points x n_features`).
        constraints:
          Optional MLB triples `(i, j, k)` as document indices.
          iHAC evaluates them with row-equivalence expansion semantics internally.
        figsize:
          Matplotlib figure size used by plotting helpers.

        Notes
        -----
        For constraint checks, iHAC uses lazy class-level semantics:
        each constrained point is interpreted together with all row-equivalent points
        (identical feature rows). This preserves the semantics of full cartesian
        constraint expansion without materializing expanded triples.
        """
        super().__init__(figsize=figsize)
        self.X = X
        self.n = X.shape[0]
        self.constraints: List[ConstraintTriple] = (
            [tuple(map(int, c)) for c in constraints] if constraints else []
        )
        self._validate_constraints_in_bounds()
        self.row_equiv_members: Dict[int, Tuple[int, ...]] = self._build_row_equivalence_members()
        self.constraint_equiv_members: List[EquivConstraintTriple] = (
            self._build_constraint_equiv_members()
        )

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
        self.merged_cluster_members_steps: List[Set[int]] = []
        self.snapshots = []
        self.reduced_snapshots = []
        self.initial_representative_points: List[int] = []
        self.initial_cluster_ids: List[int] = []
        self.cluster_rep_members: Dict[int, Set[int]] = {}
        self.cluster_rep_sizes: Dict[int, int] = {}

        self._initialize_clusters()

        # Map active cluster -> constraints involving at least one member point.
        # This lets violation checks inspect only relevant constraints for a candidate merge.
        self.cluster_to_constraint_idxs: Dict[int, Set[int]] = (
            self._index_constraints_by_active_cluster()
        )

    def _validate_constraints_in_bounds(self) -> None:
        """Validate that all constraint point indices reference existing rows in `X`."""
        for a, b, c in self.constraints:
            for point_idx in (a, b, c):
                if point_idx < 0 or point_idx >= self.n:
                    raise ValueError(
                        f"Constraint point index {point_idx} out of bounds for n={self.n}."
                    )

    def _build_row_equivalence_members(self) -> Dict[int, Tuple[int, ...]]:
        """
        Group points by identical feature rows and map each point to its full group.

        The mapping enables lazy class-level MLB checks without materializing the full
        cartesian expansion of equivalent-point constraints.
        """
        groups: Dict[bytes, List[int]] = {}
        for idx in range(self.n):
            groups.setdefault(self.X[idx].tobytes(), []).append(idx)
        members_by_point: Dict[int, Tuple[int, ...]] = {}
        for members in groups.values():
            ordered = tuple(sorted(members))
            for point_idx in ordered:
                members_by_point[point_idx] = ordered
        return members_by_point

    def _build_constraint_equiv_members(self) -> List[EquivConstraintTriple]:
        """
        Precompute row-equivalent member tuples for each base MLB triple.

        Returns
        -------
        list[EquivConstraintTriple]
          For each base `(a, b, c)` triple, stores
          `(row_equiv(a), row_equiv(b), row_equiv(c))`.
        """
        return [
            (
                self.row_equiv_members[a],
                self.row_equiv_members[b],
                self.row_equiv_members[c],
            )
            for (a, b, c) in self.constraints
        ]

    def _index_constraints_by_active_cluster(self) -> Dict[int, Set[int]]:
        """
        Build an inverted index from active cluster id to relevant constraint indices.

        A constraint is considered relevant to a cluster if the cluster currently contains
        at least one point from any row-equivalence class involved in that constraint.
        """
        cluster_to_constraint_idxs: Dict[int, Set[int]] = {
            cid: set() for cid in self.active_clusters
        }
        for idx, (members_a, members_b, members_c) in enumerate(self.constraint_equiv_members):
            touched_points = set(members_a) | set(members_b) | set(members_c)
            for point_idx in touched_points:
                cid = self.point_to_cluster[point_idx]
                cluster_to_constraint_idxs[cid].add(idx)
        return cluster_to_constraint_idxs

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
        Build initial clusters by compressing identical feature rows.

        All documents with identical topic-incidence vectors are merged into one initial
        active cluster, including documents touched by constraints. Constraint semantics
        are still enforced lazily at class level via row-equivalence sets.
        """
        groups: Dict[bytes, List[int]] = {}
        for idx in range(self.n):
            key = self.X[idx].tobytes()
            groups.setdefault(key, []).append(idx)

        for members in groups.values():
            self._add_initial_cluster(members[0], set(members))

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

    def _mapped_cluster_id(self, cluster_id: int, merge_a: int, merge_b: int) -> int:
        """
        Map old cluster IDs under a hypothetical merge.

        If `cluster_id` is either merge operand, it maps to a shared placeholder
        id (`-1`) that represents the post-merge cluster.

        Args:
            cluster_id: Cluster ID to map.
            merge_a: First cluster being merged.
            merge_b: Second cluster being merged.
        """
        if cluster_id == merge_a or cluster_id == merge_b:
            return -1
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
        count = 0
        for idx in self._relevant_constraints_for_merge(merge_a, merge_b):
            # skip already violated constraints
            if idx in self.violated_constraints:
                continue
            if self._is_constraint_violated(idx, merge_a=merge_a, merge_b=merge_b):
                count += 1
        return count

    def _relevant_constraints_for_merge(self, merge_a: int, merge_b: int) -> Set[int]:
        """
        Return constraint indices potentially affected by merging `merge_a` and `merge_b`.
        """
        return self.cluster_to_constraint_idxs.get(
            merge_a, set()
        ) | self.cluster_to_constraint_idxs.get(merge_b, set())

    @staticmethod
    def _has_distinct_partner(values: Set[int], anchor: int) -> bool:
        """Return True if `values` contains at least one element different from `anchor`."""
        return any(value != anchor for value in values)

    def _is_constraint_violated(
        self,
        constraint_idx: int,
        *,
        merge_a: Optional[int] = None,
        merge_b: Optional[int] = None,
    ) -> bool:
        """
        Check one MLB constraint under current clustering or a hypothetical merge.

        The check is class-level: each base point in a triple is replaced by all points
        with identical feature rows. This is equivalent to explicit cartesian expansion
        of constraints, but evaluated lazily.
        """
        members_a, members_b, members_c = self.constraint_equiv_members[constraint_idx]
        clusters_a = self._cluster_ids_for_members(members_a, merge_a=merge_a, merge_b=merge_b)
        clusters_b = self._cluster_ids_for_members(members_b, merge_a=merge_a, merge_b=merge_b)
        clusters_c = self._cluster_ids_for_members(members_c, merge_a=merge_a, merge_b=merge_b)
        return self._violates_expanded_semantics(clusters_a, clusters_b, clusters_c)

    def _cluster_ids_for_members(
        self,
        members: Tuple[int, ...],
        *,
        merge_a: Optional[int] = None,
        merge_b: Optional[int] = None,
    ) -> Set[int]:
        """
        Collect cluster IDs for a tuple of point indices, optionally after a hypothetical merge.
        """
        cluster_ids: Set[int] = set()
        for point_idx in members:
            cluster_ids.add(
                self._cluster_id_for_point(
                    point_idx,
                    merge_a=merge_a,
                    merge_b=merge_b,
                )
            )
        return cluster_ids

    def _cluster_id_for_point(
        self,
        point_idx: int,
        *,
        merge_a: Optional[int] = None,
        merge_b: Optional[int] = None,
    ) -> int:
        """
        Return the cluster id for one point, with optional merge projection.
        """
        cluster_id = self.point_to_cluster[point_idx]
        if merge_a is None or merge_b is None:
            return cluster_id
        return self._mapped_cluster_id(cluster_id, merge_a, merge_b)

    def _violates_expanded_semantics(
        self, clusters_a: Set[int], clusters_b: Set[int], clusters_c: Set[int]
    ) -> bool:
        """
        Evaluate the expanded MLB violation rule from class-level cluster sets.

        Equivalent expanded rule:
        `exists a in A, b in B, c in C: cluster(a) != cluster(b)` and
        `(cluster(c) == cluster(a) or cluster(c) == cluster(b))`.
        """
        for cluster_a in clusters_a:
            if cluster_a in clusters_c and self._has_distinct_partner(clusters_b, cluster_a):
                return True
        for cluster_b in clusters_b:
            if cluster_b in clusters_c and self._has_distinct_partner(clusters_a, cluster_b):
                return True
        return False

    def _update_violated_constraints(self) -> None:
        """Mark constraints violated by the current partition (class-level, lazy)."""
        for idx in range(len(self.constraints)):
            # no need to double-check already violated constraints
            if idx in self.violated_constraints:
                continue
            if self._is_constraint_violated(idx):
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
        # Persist the cluster formed at this merge step for downstream context export.
        self.merged_cluster_members_steps.append(set(new_cluster))
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

    def merge_step_context_matrix(self) -> np.ndarray:
        """
        Build a document-by-merge-step incidence matrix from merge history.

        Rows correspond to original document indices (`0..n-1`), columns correspond to
        merge steps (`1..m`). Entry `(doc, step)` is 1 iff the document belongs to the
        cluster formed at that merge step.
        """
        n_steps = len(self.merged_cluster_members_steps)
        context = np.zeros((self.n, n_steps), dtype=np.uint8)
        for step_idx, merged_members in enumerate(self.merged_cluster_members_steps):
            if merged_members:
                context[list(merged_members), step_idx] = 1
        return context

    def save_merge_step_context_csv(
        self,
        out_path: Path,
        *,
        filename: str = "merge_step_context.csv",
        object_names: Optional[Sequence[str]] = None,
    ) -> Path:
        """
        Save merge-step incidence context as CSV.

        CSV schema:
        - first column: `object`
        - following columns: `merge_step_1 ... merge_step_m`
        - values: 0/1 membership in the cluster created at each merge step.
        """
        out_path.mkdir(parents=True, exist_ok=True)
        context = self.merge_step_context_matrix()
        if object_names is None:
            row_names = [str(i) for i in range(self.n)]
        else:
            if len(object_names) != self.n:
                raise ValueError(
                    f"Expected {self.n} object names, got {len(object_names)}."
                )
            row_names = [str(name) for name in object_names]

        csv_path = out_path / filename
        header = ["object"] + [
            f"merge_step_{step_idx}" for step_idx in range(1, context.shape[1] + 1)
        ]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for obj_idx, obj_name in enumerate(row_names):
                writer.writerow([obj_name] + context[obj_idx, :].astype(int).tolist())
        logging.info("Saved merge-step context CSV: %s", csv_path)
        return csv_path

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
