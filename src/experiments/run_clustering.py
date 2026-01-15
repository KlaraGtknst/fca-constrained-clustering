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

def display_ac(data) -> None:
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
    ac.display_dendrogram(
        data,
        method="ward",
        title="Hierarchical Agglomerative Clustering Dendrogram",
    )


def display_ihac(X, constraints, out_path: Path) -> None:
    ihac = iHAC(X, constraints)
    clusters = ihac.run()
    ihac_steps = ihac.clustering_steps()
    logging.info("Clustering steps: %s", ihac_steps)
    logging.info("Final clusters: %s", clusters)

    for step_idx, clusters in enumerate(ihac_steps, start=1):
        ihac.plot_partition(X, constraints, clusters, out_path, step_idx)

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

    out_path = Path("results/ihac")
    out_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Sample data
    # ------------------------------------------------------------------
    # Load Iris data
    iris = load_iris()
    data = iris.data
    x,y = data[:, 0], data[:, 1]
    labels = iris.target

    # ------------------------------------------------------------------
    # AC clustering
    # ------------------------------------------------------------------
    display_ac(data)
    
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

    display_ihac(X, constraints, out_path)

    
