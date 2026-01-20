import logging
from pathlib import Path
import sys
import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from fcapy import context
from fcapy.lattice import ConceptLattice
from fcapy.visualizer import LineVizNx
import seaborn as sns

logger = logging.getLogger(__name__)
from clustering.ihac import iHAC

# run with Pycharm bc realtive imports in VScode give me hell:)


def read_ctx_from_json(path: Path) -> tuple[context.FormalContext, pd.DataFrame]:
    """
    Reads an FCA context from a JSON file saved in pandas 'split' orientation.
    Returns an fcapy.Context object.
    """
    df = pd.read_json(path, orient="split", compression="infer")
    bool_df = df.astype(bool)
    # fcapy expects string labels for objects/attributes.
    bool_df.index = bool_df.index.astype(str)
    bool_df.columns = bool_df.columns.astype(str)
    logger.info("%s", bool_df)
    c = context.FormalContext.from_pandas(bool_df)
    return c, bool_df


if __name__ == "__main__":
    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ctx_path = Path("../../resources/banksearch/fca_context.json")
    assert ctx_path.exists(), f"Path does not exist: {ctx_path}"
    save_path = Path("../../results/ihac")
    save_path.mkdir(parents=True, exist_ok=True)

    ctx, bool_df = read_ctx_from_json(ctx_path)
    logger.info(
        f"Loaded context with {ctx.n_objects} objects and {ctx.n_attributes} attributes."
    )

    # Run iHAC on the boolean feature matrix (0/1).
    X = bool_df.astype(int).to_numpy()
    ihac = iHAC(X)
    logger.info("Starting iHAC clustering...")
    clusters = ihac.run()
    logger.info(f"iHAC finished with {len(clusters)} clusters.")
    # save clustering steps (i.e., dendogram levels) to json: Num_clusters -> clusters
    save_path = Path("results/ihac")
    save_path.mkdir(parents=True, exist_ok=True)
    partitions = ihac.clustering_steps()
    clustering_steps = {
        len(partition): [list(cluster) for cluster in partition]
        for partition in ihac.clustering_steps()
        for partition in partitions
    }
    with open(save_path / "banksearch_ihac_steps.json", "w") as f:
        json.dump(clustering_steps, f)
    logger.info(
        f"Saved iHAC clustering steps to {save_path / 'banksearch_ihac_steps.json'}"
    )

    ihac.save_scatter_series_from_partitions(
        X,
        partitions,
        save_path,
        dataset_name="banksearch",
        method_name="ihac",
        gif_name="clustering.gif",
    )

    ## Takes time to compute the full lattice; do last
    # visualize the context's concept lattice
    lattice = ConceptLattice.from_context(ctx)
    viz = LineVizNx(lattice)
    viz.draw()
    plt.savefig(save_path / "banksearch_lattice.png")
    plt.show()
    logger.info(
        f"Saved lattice visualization to {save_path / 'banksearch_lattice.png'}"
    )
