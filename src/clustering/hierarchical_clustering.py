from __future__ import annotations

import logging
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Set, Tuple

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA


logger = logging.getLogger(__name__)
# Ward: Minimizes the total within-cluster variance. https://www.simboli.eu/m/HAC_clustering_python (14.01.2026)
LinkageType = Literal["ward", "complete", "average", "single"]
MetricType = Literal["euclidean", "l1", "l2", "manhattan", "cosine"]


class BaseClusteringWrapper(ABC):
    """
    Abstract base class for clustering wrappers.
    """

    def __init__(self, figsize: tuple[int, int] = (10, 6)) -> None:
        self.figsize = figsize

    @staticmethod
    def _labels_from_partition(
        partition: Sequence[Set[int]], n_samples: int
    ) -> np.ndarray:
        labels = np.zeros(n_samples, dtype=int)
        for cluster_idx, cluster in enumerate(partition):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    @staticmethod
    def _ensure_2d(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if data.shape[1] < 2:
            raise ValueError("Need at least 2 dimensions for scatter plotting.")
        elif data.shape[1] > 2:
            # PCA is deterministic and much cheaper than t-SNE for iterative plotting.
            logger.info("Reducing data to 2D using PCA for plotting.")
            reduced_data = PCA(n_components=2, random_state=42).fit_transform(data)
            return reduced_data[:, 0], reduced_data[:, 1]
        return data[:, 0], data[:, 1]

    def display_dendrogram(
        self,
        data: np.ndarray,
        *,
        method: str = "ward",
        metric: str = "euclidean",
        title: Optional[str] = None,
        dataset_name: Optional[str] = None,
        method_name: Optional[str] = None,
    ) -> None:
        """
        Display a hierarchical clustering dendrogram.

        :param data: Input data (n_samples, n_features)
        :param method: Linkage method (ward, single, complete, average, etc.)
        :param metric: Distance metric
        :param title: Optional plot title
        :param figsize: Figure size

        -----------
        https://www.w3schools.com/python/python_ml_hierarchial_clustering.asp (14.01.2026)
        """
        logger.info("Generating dendrogram (method=%s, metric=%s)", method, metric)

        linkage_matrix = linkage(data, method=method, metric=metric)
        logger.info(f"Linkage for dendogram: {linkage_matrix}")

        plt.figure()
        dendrogram(linkage_matrix)
        plt.xlabel("Sample index")
        plt.ylabel("Distance")

        if title:
            plt.title(f"{dataset_name} - {method_name} - {title}")
        else:
            plt.title(f"{dataset_name} - {method_name} - Dendrogram")

        plt.tight_layout()
        plt.show()

    def save_dendrogram_svg(
        self,
        data: np.ndarray,
        out_path: Path,
        *,
        dataset_name: str,
        method_name: str,
        filename: str = "dendrogram.svg",
        method: str = "ward",
        metric: str = "euclidean",
        title: Optional[str] = None,
    ) -> Path:
        """
        Save a dendrogram as SVG.

        Args:
            data: Input data (n_samples, n_features).
            out_path: Output directory for the SVG.
            filename: Output SVG filename.
            method: Linkage method.
            metric: Distance metric.
            title: Optional plot title.
        """
        out_path.mkdir(parents=True, exist_ok=True)
        linkage_matrix = linkage(data, method=method, metric=metric)
        plt.figure(figsize=self.figsize)
        dendrogram(linkage_matrix)
        plt.xlabel("Sample index")
        plt.ylabel("Distance")
        if title and dataset_name and method_name:
            plt.title(f"{dataset_name} - {method_name} - {title}")
        elif title:
            plt.title(title)
        elif dataset_name and method_name:
            plt.title(f"{dataset_name} - {method_name} - Dendrogram")
        plt.tight_layout()
        svg_path = out_path / f"{dataset_name}_{method_name}_{filename}"
        plt.savefig(svg_path, format="svg")
        plt.close()
        logger.info("Saved dendrogram: %s", svg_path)
        return svg_path

    def plot_dendrogram(self, model, **kwargs) -> None:
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
        plt.show()

    def _save_scatter(
        self,
        x: np.ndarray,
        y: np.ndarray,
        labels: np.ndarray,
        svg_path: Path,
        title: str,
        *,
        constraints: Optional[Sequence[Tuple[int, int, int]]] = None,
    ) -> None:
        """
        Save one scatter frame for a single clustering state.

        Uses a dynamic discrete colormap sized to the number of clusters in
        this frame to keep cluster colors distinctive.
        """
        plt.figure(figsize=self.figsize)
        unique_labels = sorted(set(labels.tolist()))
        n_colors = max(len(unique_labels), 2)
        # Dynamic discrete palette: avoids tab10 color reuse for >10 clusters.
        cmap = plt.get_cmap("gist_ncar", n_colors)
        label_to_idx = {label_id: idx for idx, label_id in enumerate(unique_labels)}
        color_idx = np.array([label_to_idx[label_id] for label_id in labels], dtype=float)
        scatter = plt.scatter(
            x=x,
            y=y,
            c=color_idx,
            cmap=cmap,
            vmin=0,
            vmax=n_colors - 1,
        )
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(title)
        for idx, (x_val, y_val) in enumerate(zip(x, y)):
            plt.text(x_val, y_val, f"$x_{{{idx}}}$", fontsize=7, ha="left", va="bottom")
        handles = []
        for label_id in unique_labels:
            color_pos = label_to_idx[label_id]
            handles.append(
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    color=scatter.cmap(scatter.norm(color_pos)),
                    label=f"Cluster {label_id}",
                )
            )
        if constraints:
            handles.append(
                plt.Line2D(
                    [],
                    [],
                    linestyle="",
                    label=f"Constraints:",
                )
            )
            for a, b, c in constraints:
                handles.append(
                    plt.Line2D(
                        [],
                        [],
                        linestyle="",
                        label=f"$x_{{{a}}}, x_{{{b}}}, x_{{{c}}}$",
                    )
                )
        if handles:
            plt.legend(
                handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1, 1)
            )
        plt.tight_layout()
        for format in ["svg", "png"]:
            plt.savefig(svg_path.with_suffix(f".{format}"), format=format)
        plt.close()

    def display_clustering(
        self,
        x: np.ndarray,
        y: np.ndarray,
        labels: np.ndarray,
        *,
        dataset_name: str,
        method_name: str,
        title: str = "Clustering",
    ) -> None:
        """Display a scatter plot for a clustering assignment."""
        plt.figure(figsize=self.figsize)
        unique_labels = sorted(set(labels.tolist()))
        n_colors = max(len(unique_labels), 2)
        cmap = plt.get_cmap("gist_ncar", n_colors)
        label_to_idx = {label_id: idx for idx, label_id in enumerate(unique_labels)}
        color_idx = np.array([label_to_idx[label_id] for label_id in labels], dtype=float)
        scatter = plt.scatter(
            x=x,
            y=y,
            c=color_idx,
            cmap=cmap,
            vmin=0,
            vmax=n_colors - 1,
        )
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"{dataset_name} - {method_name} - {title}")
        for idx, (x_val, y_val) in enumerate(zip(x, y)):
            plt.text(x_val, y_val, f"$x_{{{idx}}}$", fontsize=7, ha="left", va="bottom")
        handles = []
        for label_id in unique_labels:
            color_pos = label_to_idx[label_id]
            handles.append(
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    color=scatter.cmap(scatter.norm(color_pos)),
                    label=f"Cluster {label_id}",
                )
            )
        if handles:
            plt.legend(
                handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1, 1)
            )
        plt.tight_layout()
        plt.show()

    def save_scatter_series_from_labels(
        self,
        data: np.ndarray,
        labels_per_level: Dict[int, np.ndarray],
        out_path: Path,
        *,
        dataset_name: str,
        method_name: str,
        filename_prefix: str = "clustering",
        title_prefix: str = "Clustering",
        gif_name: str = "clustering.gif",
        gif_duration_ms: int = 600,
        constraints: Optional[Sequence[Tuple[int, int, int]]] = None,
    ):
        """
        Save scatter plots for each clustering level and one animated GIF.

        Args:
            data: Input data (n_samples, n_features).
            labels_per_level: Mapping from level to labels array.
            out_path: Output directory for plots.
            filename_prefix: Prefix for SVG/PNG filenames.
            title_prefix: Prefix for plot titles.
            gif_name: Filename for the GIF.
            gif_duration_ms: Duration per frame in milliseconds.

        Notes:
            - frame colors are chosen dynamically per level
            - GIF is encoded with infinite looping (`loop=0`)
        """
        out_path.mkdir(parents=True, exist_ok=True)
        x, y = self._ensure_2d(data)
        paths: List[Path] = []
        png_paths: List[Path] = []
        tmp_root = Path("tmp/cache")
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix="plots_", dir=tmp_root))
        for level in sorted(labels_per_level.keys()):
            labels = labels_per_level[level]
            path = (
                out_path
                / f"{dataset_name}_{method_name}_{filename_prefix}_{level:03d}.png"
            )
            self._save_scatter(
                x,
                y,
                labels,
                path,
                title=f"{dataset_name} - {method_name} - {title_prefix} (level {level})",
                constraints=constraints,
            )
            logger.info("Saved scatter plot to %s", path)
            png_paths.append(path)

        if png_paths:
            images = [Image.open(path) for path in png_paths]
            # Convert to indexed mode so loop metadata is respected consistently.
            gif_frames = [img.convert("P", palette=Image.Palette.ADAPTIVE) for img in images]
            gif_path = out_path / f"{dataset_name}_{method_name}_{gif_name}"
            gif_frames[0].save(
                gif_path,
                save_all=True,
                append_images=gif_frames[1:],
                duration=gif_duration_ms,
                loop=0,
                optimize=False,
                disposal=2,
            )
            for image in images:
                image.close()
            for frame in gif_frames:
                frame.close()
            for path in png_paths:
                path.unlink(missing_ok=True)
            tmp_dir.rmdir()
            logger.info("Saved GIF: %s", gif_path)
        else:
            tmp_dir.rmdir()

    def save_scatter_series_from_partitions(
        self,
        data: np.ndarray,
        partitions: Sequence[Sequence[Set[int]]],
        out_path: Path,
        *,
        dataset_name: str,
        method_name: str,
        filename_prefix: str = "clustering",
        title_prefix: str = "Clustering",
        gif_name: str = "clustering.gif",
        gif_duration_ms: int = 600,
        constraints: Optional[Sequence[Tuple[int, int, int]]] = None,
    ) -> List[Path]:
        """
        Save scatter plots (SVG) for each clustering step and a GIF over steps.

        Args:
            data: Input data (n_samples, n_features).
            partitions: List of partitions per step.
            out_path: Output directory for plots.
            filename_prefix: Prefix for SVG/PNG filenames.
            title_prefix: Prefix for plot titles.
            gif_name: Filename for the GIF.
            gif_duration_ms: Duration per frame in milliseconds.
        """
        labels_per_level = {
            idx + 1: self._labels_from_partition(partition, data.shape[0])
            for idx, partition in enumerate(partitions)
        }
        return self.save_scatter_series_from_labels(
            data,
            labels_per_level,
            out_path,
            dataset_name=dataset_name,
            method_name=method_name,
            filename_prefix=filename_prefix,
            title_prefix=title_prefix,
            gif_name=gif_name,
            gif_duration_ms=gif_duration_ms,
            constraints=constraints,
        )

    @abstractmethod
    def cluster(self, data: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Perform clustering and return cluster labels.
        """
        raise NotImplementedError


class AgglomerativeClusteringWrapper(BaseClusteringWrapper):
    """
    Hierarchical Agglomerative Clustering (HAC) wrapper.
    """

    def __init__(
        self,
        *,
        n_clusters: int = 2,
        linkage: LinkageType = "ward",
        metric: MetricType = "euclidean",
    ) -> None:
        """
        :param n_clusters: Number of clusters to form
        :param linkage: Linkage criterion (ward, complete, average, single)
        :param metric: Distance metric (ignored for ward)
        """
        super().__init__()
        self.cache_path = Path("tmp/cache")
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric

    def cluster(self, data: np.ndarray) -> Dict[int, np.ndarray]:
        logger.info(
            "Running HAC (n_clusters=%d, linkage=%s, metric=%s)",
            self.n_clusters,
            self.linkage,
            self.metric,
        )

        cluster_args = {"memory": str(self.cache_path), "compute_full_tree": True}
        labels_per_level = {}

        for n_clusters in range(1, self.n_clusters + 1):
            cluster_args.update({"n_clusters": n_clusters})

            if self.linkage == "ward":
                model = AgglomerativeClustering(
                    **cluster_args,
                    linkage="ward",
                )
            else:
                model = AgglomerativeClustering(
                    **cluster_args,
                    linkage=self.linkage,
                    metric=self.metric,
                )

            cluster_labels = model.fit_predict(data)
            labels_per_level[n_clusters] = cluster_labels

        return labels_per_level
