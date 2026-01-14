from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris

logger = logging.getLogger(__name__)
# Ward: Minimizes the total within-cluster variance. https://www.simboli.eu/m/HAC_clustering_python (14.01.2026)
LinkageType = Literal["ward", "complete", "average", "single"]
MetricType = Literal["euclidean", "l1", "l2", "manhattan", "cosine"]



class BaseClusteringWrapper(ABC):
    """
    Abstract base class for clustering wrappers.
    """
    def __init__(self, figsize: tuple[int, int]=(10, 6)):
        self.figsize = figsize,

    def display_dendrogram(
        self,
        data: np.ndarray,
        *,
        method: str = "ward",
        metric: str = "euclidean",
        title: Optional[str] = None,
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
            plt.title(title)

        plt.tight_layout()
        plt.show()

    def plot_dendrogram(self, model, **kwargs):
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

    def display_clustering(self, x: np.ndarray, y: np.ndarray, labels:List[int], title:str="Clustering"):
        plt.figure()
        plt.scatter(x=x,y=y,c=labels)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(title)

        plt.tight_layout()
        plt.show()



    @abstractmethod
    def cluster(self, data: np.ndarray) -> Dict[int,np.ndarray]:
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

    def cluster(self, data: np.ndarray) -> Dict[int,np.ndarray]:
        logger.info(
            "Running HAC (n_clusters=%d, linkage=%s, metric=%s)",
            self.n_clusters,
            self.linkage,
            self.metric,
        )

        cluster_args = {"memory":str(self.cache_path), "compute_full_tree":True}
        labels_per_level = {}
        
        for n_clusters in range(1, self.n_clusters + 1):
            cluster_args.update({"n_clusters":n_clusters})

            if self.linkage == "ward":
                model = AgglomerativeClustering(
                    **cluster_args,
                    linkage="ward",
                )
            else:
                model = AgglomerativeClustering(
                    **cluster_args,
                    linkage=self.linkage,
                    metric=self.metric,)
                
            cluster_labels = model.fit_predict(data)
            labels_per_level[n_clusters] = cluster_labels

        return labels_per_level


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Logging configuration
    # ------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ------------------------------------------------------------------
    # Sample data
    # ------------------------------------------------------------------
    # x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
    # y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

    # data = np.column_stack((x, y))

    # Load Iris data
    iris = load_iris()
    data = iris.data
    x,y = data[:, 0], data[:, 1]
    labels = iris.target

    # ------------------------------------------------------------------
    # AC clustering
    # ------------------------------------------------------------------
    ac = AgglomerativeClusteringWrapper(
        n_clusters=2,
        linkage="ward",
    )

    hier_labels = ac.cluster(data)
    logger.info(f"hier labels: {hier_labels}")
    lowest_level_labels = hier_labels[max(hier_labels.keys())]
    ac.display_clustering(x=x,y=y,labels=lowest_level_labels)
    logger.info("Cluster labels: %s", labels)

    # ------------------------------------------------------------------
    # Dendrogram
    # ------------------------------------------------------------------
    # ac.display_dendrogram(
    #     data,
    #     method="ward",
    #     title="Hierarchical Agglomerative Clustering Dendrogram",
    # )

    # ---- Dendrogram ----
    model_full = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model_full.fit(data)

    ac.plot_dendrogram(model_full, truncate_mode="level", p=3)
