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
    Load a formal context stored as pandas split-JSON.

    The experiment stores context data as a dataframe serialized with `orient="split"`.
    This helper restores that dataframe, enforces boolean incidence values, and normalizes
    row/column labels to Python `str` so the resulting dataframe can be consumed by `fcapy`.

    Parameters
    ----------
    path:
      Path to a pandas split-JSON file with `index`, `columns`, and `data`.

    Returns
    -------
    tuple[context.FormalContext, pd.DataFrame]
      1) The `fcapy` formal context created from the normalized dataframe.
      2) The normalized boolean dataframe used to create the context.
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
    logger,
    ihac: iHAC,
    out_root: Path,
    data_kind: str = "topic_model",
    save_plots: bool = True,
    object_names: list[str] | None = None,
) -> Path:
    """
    Execute iHAC and persist clustering artifacts for one dataset variant.

    Persisted artifacts include:
    - Active-cluster partitions for each merge step as JSON.
    - Document-by-merge-step context CSV (attributes = merge steps).
    - Stepwise dendrogram SVG.
    - Optional scatter-series frames and an animation generated from active partitions.

    Notes
    -----
    `iHAC` internally compresses unconstrained duplicate points. Therefore, exported
    partitions/plots operate on active representative clusters rather than the full
    object list at each step.
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

    ihac.save_merge_step_context_csv(
        save_path,
        filename=f"{data_kind}_banksearch_merge_step_context.csv",
        object_names=object_names,
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
    """
    Resolve a raw label token to the concrete topic name used in context columns.

    This is primarily needed when constraints contain labels that differ from context
    headers (for example, `C/C++` in constraints versus `C` in the context).
    """
    alias_map = {
        "C/C++": "C",
    }
    if label in available_topics:
        return label
    if label in alias_map and alias_map[label] in available_topics:
        return alias_map[label]
    split_alias = label.split("/")[0].strip()
    if split_alias in available_topics:
        return split_alias
    return label


def _build_label_to_doc_idxs(
    bool_df: pd.DataFrame, obj_to_idx: dict[str, int], hierarchy_dict: dict[str, list[str]]
) -> dict[str, list[int]]:
    """
    Build a lookup from labels to document-index candidates.

    The returned mapping supports:
    - Direct topic labels from context columns.
    - Optional hierarchy node labels (e.g. parent categories), resolved to the union
      of descendant leaf-topic documents.
    - Topic aliases normalized by `_resolve_topic_alias`.

    Parameters
    ----------
    bool_df:
      Document-by-topic incidence dataframe.
    obj_to_idx:
      Mapping from document ID to integer index used by iHAC.
    hierarchy_dict:
      Optional hierarchy mapping from parent label to child labels. An empty dict is
      valid and means "direct topics only".
    """
    available_topics = set(bool_df.columns.astype(str))
    topic_to_doc_idxs = {}
    for topic in bool_df.columns.astype(str):
        doc_ids = bool_df.index[bool_df[topic]].astype(str).tolist()
        topic_to_doc_idxs[topic] = sorted(obj_to_idx[doc_id] for doc_id in doc_ids)

    @lru_cache(maxsize=None)
    def expand_to_leaf_topics(label: str) -> tuple[str, ...]:
        # Cache recursive expansions to avoid repeatedly traversing the same subtree.
        label = _resolve_topic_alias(label, available_topics)
        if label in hierarchy_dict:
            leaves = []
            for child in hierarchy_dict[label]:
                leaves.extend(expand_to_leaf_topics(child))
            return tuple(sorted(set(leaves)))
        return (label,)

    label_to_doc_idxs = {}
    for topic, idxs in topic_to_doc_idxs.items():
        label_to_doc_idxs[topic] = idxs
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
    for topic in list(topic_to_doc_idxs.keys()):
        alias = _resolve_topic_alias(topic, available_topics)
        if alias != topic and alias in topic_to_doc_idxs:
            label_to_doc_idxs[topic] = topic_to_doc_idxs[alias]
    return label_to_doc_idxs


def _read_constraint_lines(constraints_path: Path) -> list[tuple[str, str, str]]:
    """
    Parse a CSV-like MLB file with one triple per line.

    Expected line format: `token_x,token_y,token_z`.
    Empty lines are ignored. Malformed lines are skipped with a warning.
    """
    triples = []
    with open(constraints_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",") if part.strip()]
            if len(parts) != 3:
                logger.warning("Skipping malformed constraint line: %s", line)
                continue
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def _build_object_index_maps(obj_ids: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """
    Build forward and reverse index maps for document IDs.

    Parameters
    ----------
    obj_ids:
      Ordered document IDs from the context dataframe index.

    Returns
    -------
    tuple[dict[str, int], dict[int, str]]
      1) `obj_to_idx`: document ID -> integer row index.
      2) `idx_to_obj`: integer row index -> document ID.
    """
    obj_to_idx = {obj_id: idx for idx, obj_id in enumerate(obj_ids)}
    idx_to_obj = {idx: obj_id for obj_id, idx in obj_to_idx.items()}
    return obj_to_idx, idx_to_obj


def _invert_obj_to_idx(obj_to_idx: dict[str, int]) -> dict[int, str]:
    """
    Invert an existing `obj_to_idx` mapping.
    """
    return {idx: obj_id for obj_id, idx in obj_to_idx.items()}


def _dedupe_constraint_triples(
    triples: list[tuple[int, int, int]]
) -> list[tuple[int, int, int]]:
    """
    Remove duplicate triples while preserving source order for reproducibility.
    """
    return list(dict.fromkeys(triples))


def _load_docid_constraints(
    constraints_path: Path, obj_to_idx: dict[str, int]
) -> list[tuple[int, int, int]]:
    """
    Load a document-ID MLB file and map each token to integer indices.

    Parameters
    ----------
    constraints_path:
      Path to normalized MLB file with lines `doc_id_x,doc_id_y,doc_id_z`.
    obj_to_idx:
      Mapping from document ID to integer row index in the current context order.

    Returns
    -------
    list[tuple[int, int, int]]
      Deduplicated MLB constraints in integer-index form.
    """
    constraints: list[tuple[int, int, int]] = []
    for token_x, token_y, token_z in _read_constraint_lines(constraints_path):
        tokens = (token_x, token_y, token_z)
        if not all(token in obj_to_idx for token in tokens):
            missing = [token for token in tokens if token not in obj_to_idx]
            raise KeyError(
                "MLB constraint contains non-document IDs: " + ", ".join(missing)
            )
        constraints.append(tuple(obj_to_idx[token] for token in tokens))
    return _dedupe_constraint_triples(constraints)


def _ensure_docid_constraints_file(
    *,
    docid_constraints_path: Path,
    source_constraints_path: Path,
    hierarchy_path: Path,
    bool_df: pd.DataFrame,
    obj_to_idx: dict[str, int],
) -> None:
    """
    Ensure the doc-ID MLB constraints file exists, generating it if required.

    Resolution strategy when generating:
    1. If a token already matches a document ID, keep it directly.
    2. Otherwise resolve aliases against context columns.
    3. If still not resolvable as a direct topic, use hierarchy expansion (when needed)
       to map category labels to candidate document indices.
    4. Use deterministic representative selection (`candidate_idxs[0]`) to preserve the
       existing experiment behavior.

    The generated file always contains only document IDs, which keeps runtime parsing
    simple and removes hierarchy from the normal hot path.
    """
    if docid_constraints_path.exists():
        return
    if not source_constraints_path.exists():
        raise FileNotFoundError(
            f"Neither doc-id constraints nor source constraints exist. "
            f"Missing: {docid_constraints_path} and {source_constraints_path}"
        )

    source_triples = _read_constraint_lines(source_constraints_path)
    # Only load/consult hierarchy if at least one token is not already a document ID.
    needs_label_resolution = any(
        token not in obj_to_idx for triple in source_triples for token in triple
    )

    hierarchy_dict = {}
    if needs_label_resolution and hierarchy_path.exists():
        with open(hierarchy_path, "r", encoding="utf-8") as f:
            hierarchy_dict = json.load(f)

    label_to_doc_idxs = _build_label_to_doc_idxs(bool_df, obj_to_idx, hierarchy_dict)
    available_topics = set(bool_df.columns.astype(str))
    idx_to_obj = _invert_obj_to_idx(obj_to_idx)
    converted = []
    for triple in source_triples:
        idx_triple = []
        for token in triple:
            if token in obj_to_idx:
                idx_triple.append(obj_to_idx[token])
                continue
            resolved_token = _resolve_topic_alias(token, available_topics)
            candidate_idxs = label_to_doc_idxs.get(resolved_token, [])
            if not candidate_idxs:
                hint = ""
                if not hierarchy_path.exists():
                    hint = f" ({hierarchy_path} is missing)"
                raise KeyError(
                    f"Cannot resolve MLB token '{token}' to document IDs{hint}."
                )
            # Keep current semantics: choose first representative doc for this label.
            idx_triple.append(candidate_idxs[0])
        converted.append(tuple(idx_triple))

    # Remove duplicates while preserving original order from source constraints.
    converted = _dedupe_constraint_triples(converted)
    docid_constraints_path.parent.mkdir(parents=True, exist_ok=True)
    with open(docid_constraints_path, "w", encoding="utf-8") as f:
        for idx_x, idx_y, idx_z in converted:
            f.write(f"{idx_to_obj[idx_x]},{idx_to_obj[idx_y]},{idx_to_obj[idx_z]}\n")
    logger.info(
        "Generated doc-id MLB file with %d constraints at %s.",
        len(converted),
        docid_constraints_path,
    )


def _load_or_generate_docid_constraints(
    *,
    docid_constraints_path: Path,
    source_constraints_path: Path,
    hierarchy_path: Path,
    bool_df: pd.DataFrame,
    obj_to_idx: dict[str, int],
) -> list[tuple[int, int, int]]:
    """
    Ensure and load doc-ID MLB constraints as integer-index triples.
    """
    _ensure_docid_constraints_file(
        docid_constraints_path=docid_constraints_path,
        source_constraints_path=source_constraints_path,
        hierarchy_path=hierarchy_path,
        bool_df=bool_df,
        obj_to_idx=obj_to_idx,
    )
    return _load_docid_constraints(
        constraints_path=docid_constraints_path,
        obj_to_idx=obj_to_idx,
    )


def run_ground_truth_experiment() -> Path:
    """
    Run the ground-truth BankSearch iHAC experiment end-to-end.

    Pipeline
    --------
    1. Load document-topic context (`fca_gt_context.json`):
       rows are document IDs, columns are topic labels, values are boolean incidence entries.
    2. Build a document-ID -> integer-row mapping (`obj_to_idx`):
       iHAC constraints address rows by integer positions in `X`, not by string IDs.
    3. Ensure/load normalized doc-ID MLB constraints:
       `mlb_banksearch_docids.txt` stores one triple per line (`doc_id_x,doc_id_y,doc_id_z`),
       then each token is mapped to an integer triple `(i, j, k)` via `obj_to_idx`.
    4. Build feature matrix `X` from the original context dataframe:
       keep document IDs as labels for FCA readability, but pass `X = bool_df.astype(int).to_numpy()`
       to iHAC so row order matches the integer constraints.
    5. Run iHAC and persist artifacts (steps JSON, dendrogram SVG, optional plots/GIF).

    Returns
    -------
    Path
      Directory where iHAC output artifacts were written.
    """
    # ground truth context and constraints
    # Document-topic context in pandas split-JSON format:
    # - `columns`: topic labels
    # - `index`: document IDs
    # - `data`: boolean incidence matrix (doc has topic)
    gt_ctx_path = PROJECT_ROOT / "resources/banksearch/ground_truth/fca_gt_context.json"
    assert gt_ctx_path.exists(), f"Path does not exist: {gt_ctx_path}"

    # Normalized MLB constraints used at runtime:
    # each line is `doc_id_x,doc_id_y,doc_id_z`.
    mlb_gt_constraints_docids_path = (
        PROJECT_ROOT / "resources/banksearch/ground_truth/mlb_banksearch_docids.txt"
    )

    # Source MLB constraints (topic/category labels and/or doc IDs),
    # e.g. `Astronomy,Biology,Finance`.
    mlb_gt_constraints_source_path = (
        PROJECT_ROOT / "resources/banksearch/ground_truth/mlb_banksearch.txt"
    )

    # Category hierarchy for resolving parent labels (e.g. `Finance`) to leaf topics.
    # Used only when generating the doc-ID MLB file from the source constraints.
    category_hierarchy_path = (
        PROJECT_ROOT / "resources/banksearch/ground_truth/category_hierarchy.json"
    )

    # Output directory for iHAC artifacts (steps JSON, dendrogram SVG, optional plots/GIF).
    save_path = PROJECT_ROOT / "results/ihac/ground_truth/"
    save_path.mkdir(parents=True, exist_ok=True)

    # Ground-truth context with MLB constraints:
    # - Keep the context indexed by document IDs (e.g., "A0975") for readability.
    # - iHAC still consumes integer-index constraint triples tied to row positions in X.
    # - If the doc-ID MLB file is missing, generate it once from source constraints.
    gt_ctx, gt_bool_df = read_ctx_from_json(gt_ctx_path)
    gt_obj_ids = list(gt_bool_df.index)
    obj_to_idx, _ = _build_object_index_maps(gt_obj_ids)

    # `mlb_gt_constraints` contains MLB triples as integer document indices:
    #   Each index is an anchor into the original document set. Inside iHAC, each anchor
    #   is interpreted together with all row-equivalent documents (same topic row), so
    #   the constraint applies class-level even though the stored triple is base-sized.
    mlb_gt_constraints = _load_or_generate_docid_constraints(
        docid_constraints_path=mlb_gt_constraints_docids_path,
        source_constraints_path=mlb_gt_constraints_source_path,
        hierarchy_path=category_hierarchy_path,
        bool_df=gt_bool_df,
        obj_to_idx=obj_to_idx,
    )
    logger.info("Loaded %d base MLB constraints.", len(mlb_gt_constraints))
    logger.info(
        f"Loaded topic model context with {gt_ctx.n_objects} objects and {gt_ctx.n_attributes} attributes."
    )
    gt_X = gt_bool_df.astype(int).to_numpy()
    logger.info(
        "Starting iHAC clustering on ground truth context with MLB constraints..."
    )
    ihac = iHAC(X=gt_X, constraints=mlb_gt_constraints)
    return run_exp_on_data(
        logger=logger,
        ihac=ihac,
        out_root=PROJECT_ROOT,
        data_kind="ground_truth",
        save_plots=True,
        object_names=gt_obj_ids,
    )


if __name__ == "__main__":
    # Configure logger for standalone script execution.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_ground_truth_experiment()
