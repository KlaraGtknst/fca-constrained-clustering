import logging
import os
from pathlib import Path
import json
from functools import lru_cache
import pandas as pd
from fcapy import context

logger = logging.getLogger(__name__)

# Keep Matplotlib/Fontconfig caches inside workspace to avoid expensive rebuilds.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "tmp" / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / "tmp" / "xdg-cache"))
from clustering.ihac import iHAC

# run with Pycharm bc realtive imports in VScode give me hell:)


def read_ctx_from_json(path: Path) -> tuple[context.FormalContext, pd.DataFrame]:
    """
    Read FCA context from pandas split-JSON and enforce string labels.

    fcapy requires strict Python str labels for both object names and attributes.
    """
    df = pd.read_json(path, orient="split", compression="infer")
    bool_df = df.astype(bool)
    # fcapy expects string labels for objects/attributes.
    bool_df.index = bool_df.index.astype(str)
    bool_df.columns = bool_df.columns.astype(str)
    logger.info(
        "Loaded context dataframe with shape=%s, first columns=%s",
        bool_df.shape,
        list(bool_df.columns[:5]),
    )
    c = context.FormalContext.from_pandas(bool_df)
    return c, bool_df


def run_exp_on_data(
    logger, ihac: iHAC, out_root: Path, data_kind: str = "topic_model", save_plots: bool = True
) -> Path:
    """
    Run iHAC and persist compressed-step artifacts.

    Optimization rationale:
    - `iHAC` now compresses unconstrained duplicate points up-front.
    - Plotting uses only active compressed clusters (representatives), not all objects.
    - The GIF frames follow only real active-cluster states after each merge.
    """
    clusters = ihac.run()
    logger.info(f"iHAC finished with {len(clusters)} clusters.")
    save_path = out_root / f"results/ihac/{data_kind}"
    save_path.mkdir(parents=True, exist_ok=True)

    # Active-cluster steps over compressed representatives.
    # Step 1 = initial active clusters; following steps are merge states.
    partitions = ihac.active_cluster_steps(include_initial=True)
    clustering_steps = {
        step_idx: [sorted(cluster) for cluster in partition]
        for step_idx, partition in enumerate(partitions, start=1)
    }
    with open(save_path / f"{data_kind}_banksearch_ihac_steps.json", "w") as f:
        json.dump(clustering_steps, f)
    logger.info(
        f"Saved iHAC clustering steps to {save_path / f'{data_kind}_banksearch_ihac_steps.json'}"
    )

    ihac.save_step_dendrogram(
        save_path,
        dataset_name="banksearch",
        method_name="ihac",
        filename=f"{data_kind}_dendrogram_steps.svg",
    )

    if save_plots:
        reduced_X = ihac.representative_data()
        ihac.save_scatter_series_from_partitions(
            reduced_X,
            partitions,
            save_path,
            dataset_name="banksearch",
            method_name="ihac",
            filename_prefix=f"{data_kind}_active_step",
            title_prefix="Active Clusters",
            gif_name=f"{data_kind}_active_clusters.gif",
        )
    else:
        logger.info(
            "Skipping active-cluster GIF rendering."
        )

    return save_path


def _resolve_topic_alias(label: str, available_topics: set[str]) -> str:
    alias_map = {
        "C/C++": "C",
    }
    if label in available_topics:
        return label
    if label in alias_map and alias_map[label] in available_topics:
        return alias_map[label]
    # fallback for labels like "C/C++" -> "C"
    split_alias = label.split("/")[0].strip()
    if split_alias in available_topics:
        return split_alias
    return label


def _build_label_to_doc_idxs(
    bool_df: pd.DataFrame, obj_to_idx: dict[str, int], hierarchy_dict: dict[str, list[str]]
) -> dict[str, list[int]]:
    """
    Map topic labels (and hierarchy nodes) to sorted document-index lists.

    Example:
    - leaf topic 'Astronomy' -> docs with Astronomy=1
    - parent 'Science' -> union of docs from leaf topics under Science
    """
    available_topics = set(bool_df.columns.astype(str))
    topic_to_doc_idxs = {}
    for topic in bool_df.columns.astype(str):
        doc_ids = bool_df.index[bool_df[topic]].astype(str).tolist()
        topic_to_doc_idxs[topic] = sorted(obj_to_idx[doc_id] for doc_id in doc_ids)

    @lru_cache(maxsize=None)
    def expand_to_leaf_topics(label: str) -> tuple[str, ...]:
        label = _resolve_topic_alias(label, available_topics)
        if label in hierarchy_dict:
            leaves = []
            for child in hierarchy_dict[label]:
                leaves.extend(expand_to_leaf_topics(child))
            return tuple(sorted(set(leaves)))
        return (label,)

    label_to_doc_idxs = {}
    # direct topics
    for topic, idxs in topic_to_doc_idxs.items():
        label_to_doc_idxs[topic] = idxs
    # hierarchy nodes (e.g. Finance, Science, Programming)
    for node in hierarchy_dict:
        leaf_topics = expand_to_leaf_topics(node)
        idxs = sorted(
            {
                idx
                for leaf in leaf_topics
                for idx in topic_to_doc_idxs.get(
                    _resolve_topic_alias(leaf, available_topics), []
                )
            }
        )
        label_to_doc_idxs[node] = idxs
    # also keep aliases (e.g. C/C++)
    for topic in list(topic_to_doc_idxs.keys()):
        alias = _resolve_topic_alias(topic, available_topics)
        if alias != topic and alias in topic_to_doc_idxs:
            label_to_doc_idxs[topic] = topic_to_doc_idxs[alias]
    return label_to_doc_idxs


if __name__ == "__main__":
    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # ground truth context and constraints
    gt_ctx_path = PROJECT_ROOT / "resources/banksearch/ground_truth/fca_gt_context.json" # stores which document covers
    # which topics
    assert gt_ctx_path.exists(), f"Path does not exist: {gt_ctx_path}"
    mlb_gt_constraints_path = PROJECT_ROOT / "resources/banksearch/ground_truth/mlb_banksearch.txt"  # stores MLB constraints
    # extracted
    # from GT/ user topic hierarchy
    assert mlb_gt_constraints_path.exists(), f"Path does not exist: {mlb_gt_constraints_path}"
    category_hierarchy_path = (
        PROJECT_ROOT / "resources/banksearch/ground_truth/category_hierarchy.json"
    )
    assert (
        category_hierarchy_path.exists()
    ), f"Path does not exist: {category_hierarchy_path}"

    # save path for results
    save_path = PROJECT_ROOT / "results/ihac/ground_truth/"
    save_path.mkdir(parents=True, exist_ok=True)

    ## ground truth context with MLB constraints
    # The context remains document-by-topic.
    # MLB file lines can contain either document IDs or topic/hierarchy labels.
    # We resolve labels to concrete document indices for iHAC triples.
    gt_ctx, gt_bool_df = read_ctx_from_json(gt_ctx_path)
    gt_obj_ids = list(gt_bool_df.index)
    obj_to_idx = {obj_id: idx for idx, obj_id in enumerate(gt_obj_ids)}
    idx_to_obj = {idx: obj_id for obj_id, idx in obj_to_idx.items()}
    with open(category_hierarchy_path, "r", encoding="utf-8") as f:
        hierarchy_dict = json.load(f)
    label_to_doc_idxs = _build_label_to_doc_idxs(gt_bool_df, obj_to_idx, hierarchy_dict)
    gt_bool_df_int = gt_bool_df.copy()
    # fcapy requires object names to be strict Python str.
    gt_bool_df_int.index = [str(obj_to_idx[obj_id]) for obj_id in gt_obj_ids]
    gt_ctx = context.FormalContext.from_pandas(gt_bool_df_int)
    # load MLB constraints
    mlb_gt_constraints = []
    with open(mlb_gt_constraints_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = [part.strip() for part in line.split(",") if part.strip()]
                if len(parts) != 3:
                    logger.warning("Skipping malformed constraint line: %s", line)
                    continue
                # Supports both document-ID constraints and topic-label constraints.
                # Topic labels are mapped to representative document indices.
                triple_candidates = []
                for token in parts:
                    if token in obj_to_idx:
                        triple_candidates.append([obj_to_idx[token]])
                        continue
                    resolved_token = _resolve_topic_alias(token, set(gt_bool_df.columns.astype(str)))
                    candidate_idxs = label_to_doc_idxs.get(resolved_token, [])
                    if not candidate_idxs:
                        raise KeyError(
                            f"Cannot resolve MLB token '{token}' to document IDs."
                        )
                    triple_candidates.append(candidate_idxs)
                mlb_gt_constraints.append(tuple(candidates[0] for candidates in triple_candidates))
    mlb_gt_constraints = list(dict.fromkeys(mlb_gt_constraints))
    logger.info("Loaded %d MLB constraints.", len(mlb_gt_constraints))
    logger.info(
        f"Loaded topic model context with {gt_ctx.n_objects} objects and {gt_ctx.n_attributes} attributes."
    )
    gt_X = gt_bool_df_int.astype(int).to_numpy()
    logger.info(
        "Starting iHAC clustering on ground truth context with MLB constraints..."
    )
    ihac = iHAC(X=gt_X, constraints=mlb_gt_constraints)
    save_path = run_exp_on_data(
        logger=logger,
        ihac=ihac,
        out_root=PROJECT_ROOT,
        data_kind="ground_truth",
        save_plots=True,
    )
