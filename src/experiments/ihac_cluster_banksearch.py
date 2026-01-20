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


def run_exp_on_data(logger, X, ihac, data_kind: str = "topic_model") -> Path:
    clusters = ihac.run()
    logger.info(f"iHAC finished with {len(clusters)} clusters.")
    # save clustering steps (i.e., dendogram levels) to json: Num_clusters -> clusters
    save_path = Path("results/ihac")
    save_path.mkdir(parents=True, exist_ok=True)
    partitions = ihac.clustering_steps()
    clustering_steps = {
        len(partition): [list(cluster) for cluster in partition]
        for partition in partitions
    }
    with open(save_path / f"{data_kind}_banksearch_ihac_steps.json", "w") as f:
        json.dump(clustering_steps, f)
    logger.info(
        f"Saved iHAC clustering steps to {save_path / f'{data_kind}_banksearch_ihac_steps.json'}"
    )

    ihac.save_scatter_series_from_partitions(
        X,
        partitions,
        save_path,
        dataset_name="banksearch",
        method_name="ihac",
        gif_name=f"{data_kind}_clustering.gif",
    )

    return save_path


if __name__ == "__main__":
    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ## paths
    # obtained from topic model
    topic_model_ctx_path = Path("../../resources/banksearch/fca_context.json")
    assert topic_model_ctx_path.exists(), f"Path does not exist: {topic_model_ctx_path}"
    # ground truth context and constraints
    gt_ctx_path = Path("../../resources/banksearch/fca_gt_context.json")
    assert gt_ctx_path.exists(), f"Path does not exist: {gt_ctx_path}"
    mlb_constraints_path = Path("../../resources/banksearch/mlb_banksearch.txt")
    assert mlb_constraints_path.exists(), f"Path does not exist: {mlb_constraints_path}"
    # save path for results
    save_path = Path("../../results/ihac")
    save_path.mkdir(parents=True, exist_ok=True)

    ## topic model context w/o constraints
    tm_ctx, tm_bool_df = read_ctx_from_json(topic_model_ctx_path)
    logger.info(
        f"Loaded topic model context with {tm_ctx.n_objects} objects and {tm_ctx.n_attributes} attributes."
    )
    # Run iHAC on the boolean feature matrix (0/1).
    tm_X = tm_bool_df.astype(int).to_numpy()
    ihac = iHAC(X=tm_X)
    logger.info("Starting iHAC clustering on topic model context w/o constraints...")
    save_path = run_exp_on_data(
        logger=logger, X=tm_X, ihac=ihac, data_kind="topic_model"
    )

    ## ground truth context with MLB constraints
    # topics are string labels; need to map to int IDs for clustering
    gt_ctx, gt_bool_df = read_ctx_from_json(gt_ctx_path)
    gt_obj_ids = list(gt_bool_df.index)
    obj_to_idx = {obj_id: idx for idx, obj_id in enumerate(gt_obj_ids)}
    idx_to_obj = {idx: obj_id for obj_id, idx in obj_to_idx.items()}
    gt_bool_df_int = gt_bool_df.copy()
    gt_bool_df_int.index = [obj_to_idx[obj_id] for obj_id in gt_obj_ids]
    gt_ctx = context.FormalContext.from_pandas(gt_bool_df_int)
    # load MLB constraints
    mlb_constraints = []
    with open(mlb_constraints_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = [part.strip() for part in line.split(",") if part.strip()]
                mlb_constraints.append(tuple(obj_to_idx[obj_id] for obj_id in parts))
    logger.info(
        f"Loaded topic model context with {gt_ctx.n_objects} objects and {gt_ctx.n_attributes} attributes."
    )
    gt_X = gt_bool_df_int.astype(int).to_numpy()
    logger.info(
        "Starting iHAC clustering on ground truth context with MLB constraints..."
    )
    ihac = iHAC(X=gt_X, constraints=mlb_constraints)
    save_path = run_exp_on_data(
        logger=logger, X=gt_X, ihac=ihac, data_kind="ground_truth"
    )

    # TODO: Compare clusterings to known labels

    ## Takes time to compute the full lattice; do last
    # visualize the context's concept lattice
    lattice = ConceptLattice.from_context(tm_ctx)
    viz = LineVizNx(lattice)
    viz.draw()
    plt.savefig(save_path / "banksearch_lattice.png")
    plt.show()
    logger.info(
        f"Saved lattice visualization to {save_path / 'banksearch_lattice.png'}"
    )
