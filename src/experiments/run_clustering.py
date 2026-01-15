import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from clustering.hierarchical_clustering import AgglomerativeClusteringWrapper
from clustering.ihac import iHAC

def display_ac(data: np.ndarray, out_path: Path, dataset_name: str) -> None:
    ac = AgglomerativeClusteringWrapper(
        n_clusters=2,
        linkage="ward",
    )

    hier_labels = ac.cluster(data)
    logger.info("hier labels: %s", hier_labels)
    lowest_level_labels = hier_labels[max(hier_labels.keys())]
    x_coords, y_coords = data[:, 0], data[:, 1]
    ac.display_clustering(
        x=x_coords,
        y=y_coords,
        labels=lowest_level_labels,
        dataset_name=dataset_name,
        method_name="HAC",
    )
    logger.info("Cluster labels: %s", lowest_level_labels)

    # ------------------------------------------------------------------
    # Dendrogram
    # ------------------------------------------------------------------
    ac.display_dendrogram(
        data,
        method="ward",
        title="Hierarchical Agglomerative Clustering Dendrogram",
        dataset_name=dataset_name,
        method_name="HAC",
    )
    ac.save_dendrogram_svg(
        data,
        out_path,
        dataset_name=dataset_name,
        method_name="HAC",
        filename="dendrogram.svg",
        method="ward",
        title="Hierarchical Agglomerative Clustering Dendrogram",
    )
    ac.save_scatter_series_from_labels(
        data,
        hier_labels,
        out_path,
        dataset_name=dataset_name,
        method_name="HAC",
        filename_prefix="level",
        gif_name="hac.gif",
    )


def display_ihac(X: np.ndarray, constraints, out_path: Path, dataset_name: str) -> None:
    ihac = iHAC(X, constraints)
    clusters = ihac.run()
    ihac_steps = ihac.clustering_steps()
    logging.info("Clustering steps: %s", ihac_steps)
    logging.info("Final clusters: %s", clusters)
    ihac.save_scatter_series_from_partitions(
        X,
        ihac_steps,
        out_path,
        dataset_name=dataset_name,
        method_name="iHAC",
        filename_prefix="step",
        gif_name="ihac.gif",
        constraints=constraints,
    )

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Logging configuration
    # ------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    hac_out_path = Path("results/hac")
    hac_out_path.mkdir(parents=True, exist_ok=True)

    ihac_out_path = Path("results/ihac")
    ihac_out_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Sample data
    # ------------------------------------------------------------------
    # Load Iris data
    iris = load_iris()
    data = iris.data
    _ = iris.target

    # ------------------------------------------------------------------
    # AC clustering
    # ------------------------------------------------------------------
    display_ac(data, hac_out_path, dataset_name="iris")
    
    # ------------------------------------------------------------------    
    # iHAC clustering with constraints
    # ------------------------------------------------------------------

    X = np.array([
        [1.0, 2.0],  # point 0
        [1.5, 1.8],  # point 1
        [5.0, 8.0],  # point 2
        [8.0, 8.0],  # point 3
        [1.2, 0.9]   # point 4
    ])
    constraints = [
        (0, 1, 2),  # merge 0 & 1 before cluster containing 2
        (0, 4, 3),  # merge 0 & 4 before cluster containing 3
    ]

    display_ihac(X, constraints, ihac_out_path, dataset_name="toy")

    
