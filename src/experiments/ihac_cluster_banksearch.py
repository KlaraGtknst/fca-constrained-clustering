import logging
import os
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
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    # ground truth context and constraints
    gt_ctx_path = PROJECT_ROOT / "resources/banksearch/ground_truth/fca_gt_context.json" # stores which document covers
    # which topics
    assert gt_ctx_path.exists(), f"Path does not exist: {gt_ctx_path}"
    mlb_gt_constraints_path = PROJECT_ROOT / "resources/banksearch/ground_truth/mlb_banksearch.txt"  # stores MLB constraints
    # extracted
    # from GT/ user topic hierarchy
    assert mlb_gt_constraints_path.exists(), f"Path does not exist: {mlb_gt_constraints_path}"

    # save path for results
    save_path = PROJECT_ROOT / "results/ihac/ground_truth/"
    save_path.mkdir(parents=True, exist_ok=True)

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
    mlb_gt_constraints = []
    with open(mlb_gt_constraints_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = [part.strip() for part in line.split(",") if part.strip()]
                mlb_gt_constraints.append(tuple(obj_to_idx[obj_id] for obj_id in parts))
    logger.info(
        f"Loaded topic model context with {gt_ctx.n_objects} objects and {gt_ctx.n_attributes} attributes."
    )
    gt_X = gt_bool_df_int.astype(int).to_numpy()
    logger.info(
        "Starting iHAC clustering on ground truth context with MLB constraints..."
    )
    ihac = iHAC(X=gt_X, constraints=mlb_gt_constraints)
    save_path = run_exp_on_data(
        logger=logger, X=gt_X, ihac=ihac, data_kind="ground_truth"
    )
