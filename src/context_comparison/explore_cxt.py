"""
Compare FCA attributes (Mi) against ground-truth topics, including Mi-combinations.

This script answers:
A) Single Mi matches:
   - Does any Mi correspond EXACTLY to one GT topic (same doc set)?
   - Does any Mi correspond EXACTLY to union/intersection of a few GT topics?

B) Mi-combination matches (NEW):
   - Does INTERSECTION of several Mi extents equal a GT topic?
   - Does UNION of several Mi extents equal a GT topic?

Inputs:
- Burmeister context (.cxt): objects(doc ids) x attributes(Mi)
- fca_gt_context.json: {"columns":[topics...], "index":[doc ids...], "data":[[bool...], ...]}

Outputs (written to results/context_comparison/MLB_GT/):
- mi_single_to_gt.json
- mi_single_to_gt.csv
- mi_combo_to_gt.json
- mi_combo_to_gt.csv
- gt_to_mi_explanations.json
- gt_topic_sizes.json
- mi_sizes.json

Usage:
    python compare_mi_to_gt_plus_combos.py \
        --cxt mlb_expanded.cxt \
        --gt fca_gt_context.json \
        --out results/context_comparison/MLB_GT \
        --max-gt-combo 3 \
        --max-mi-combo 3
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple


# ----------------------------
# Burmeister .cxt parsing
# ----------------------------

def read_burmeister_cxt(path: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Read a Burmeister FCA context (.cxt).

    Returns:
        objects: list of object ids (doc ids)
        attributes: list of attribute names (e.g., M0..M14)
        rows: list of incidence strings, length == len(objects), each string length == len(attributes)

    Notes:
        Burmeister .cxt format (typical):
            B
            <num_objects>
            <num_attributes>
            <object_name_1>
            ...
            <object_name_n>
            <attribute_1>
            ...
            <attribute_m>
            <incidence_row_1>   # '.' or 'X'
            ...
            <incidence_row_n>
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    if not lines:
        raise ValueError(f"Empty file: {path}")

    # Find header "B" (sometimes first line)
    # Minimal robust parsing: assume first non-empty is "B" or "b"
    i = 0
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    if i >= len(lines):
        raise ValueError(f"No content in file: {path}")

    header = lines[i].strip()
    if header.lower() != "b":
        raise ValueError(f"Expected Burmeister header 'B' at line {i+1}, got: {header!r}")
    i += 1

    # Next lines: n_objects, n_attributes
    def next_nonempty(idx: int) -> Tuple[int, str]:
        while idx < len(lines) and lines[idx].strip() == "":
            idx += 1
        if idx >= len(lines):
            raise ValueError("Unexpected EOF while parsing .cxt")
        return idx, lines[idx].strip()

    i, n_obj_s = next_nonempty(i)
    i += 1
    i, n_attr_s = next_nonempty(i)
    i += 1

    try:
        n_objects = int(n_obj_s)
        n_attrs = int(n_attr_s)
    except Exception as e:
        raise ValueError(f"Failed to parse object/attribute counts: {n_obj_s!r}, {n_attr_s!r}") from e

    # Read object names
    objects: List[str] = []
    for _ in range(n_objects):
        i, obj = next_nonempty(i)
        objects.append(obj)
        i += 1

    # Read attribute names
    attributes: List[str] = []
    for _ in range(n_attrs):
        i, attr = next_nonempty(i)
        attributes.append(attr)
        i += 1

    # Read incidence matrix rows
    rows: List[str] = []
    for _ in range(n_objects):
        i, row = next_nonempty(i)
        if len(row) != n_attrs:
            raise ValueError(
                f"Incidence row length mismatch at line {i+1}: "
                f"expected {n_attrs}, got {len(row)}"
            )
        rows.append(row)
        i += 1

    return objects, attributes, rows


def cxt_attribute_extents(objects: List[str], attributes: List[str], rows: List[str]) -> Dict[str, Set[str]]:
    """
    Build extent sets for each attribute from a parsed Burmeister context.

    Returns:
        dict attr -> set(objects that have attr)
    """
    attr_to_docs: Dict[str, Set[str]] = {a: set() for a in attributes}
    for obj, row in zip(objects, rows):
        for j, mark in enumerate(row):
            if mark.upper() == "X":
                attr_to_docs[attributes[j]].add(obj)
    return attr_to_docs


# ----------------------------
# GT JSON parsing
# ----------------------------

def read_gt_context_json(path: str) -> Tuple[List[str], List[str], List[List[bool]]]:
    """
    Read the ground-truth FCA context JSON.

    Expected shape:
        {
          "columns": [topic1, topic2, ...],
          "index": [doc1, doc2, ...],
          "data": [[bool,bool,...], ...]   # rows aligned with index, cols aligned with columns
        }

    Returns:
        columns, index, data
    """
    with open(path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    columns = gt["columns"]
    index = gt["index"]
    data = gt["data"]

    if len(data) != len(index):
        raise ValueError("GT 'data' row count must match 'index' length")
    for r, row in enumerate(data):
        if len(row) != len(columns):
            raise ValueError(f"GT 'data' row {r} length must match 'columns' length")

    # Ensure actual booleans
    data_bool: List[List[bool]] = [[bool(x) for x in row] for row in data]
    return columns, index, data_bool


def gt_topic_docsets(columns: List[str], index: List[str], data: List[List[bool]]) -> Dict[str, Set[str]]:
    """
    Convert GT context (columns/index/data) into doc sets per topic.

    Returns:
        dict topic -> set(doc ids that have topic)
    """
    topic_to_docs: Dict[str, Set[str]] = {c: set() for c in columns}
    for doc_id, row in zip(index, data):
        for j, has_topic in enumerate(row):
            if has_topic:
                topic_to_docs[columns[j]].add(doc_id)
    return topic_to_docs


# ----------------------------
# Matching logic
# ----------------------------

@dataclass(frozen=True)
class ComboMatch:
    """Represents an exact combination explanation for an Mi extent."""
    op: str  # "UNION" or "INTERSECTION"
    items: Tuple[str, ...]  # topics or Mi names


def exact_set_matches(target: Set[str], name_to_set: Dict[str, Set[str]]) -> List[str]:
    """Return all names in name_to_set whose set equals target."""
    return [name for name, s in name_to_set.items() if s == target]

def _apply_op(sets: List[Set[str]], op: str) -> Set[str]:
    """Apply UNION or INTERSECTION to a list of sets."""
    if not sets:
        return set()
    if op == "UNION":
        return set().union(*sets)
    if op == "INTERSECTION":
        inter = set(sets[0])
        for s in sets[1:]:
            inter.intersection_update(s)
        return inter
    raise ValueError(f"Unknown op: {op}")


def _is_candidate_nontrivial(target: Set[str], sets: List[Set[str]], op: str) -> bool:
    """
    Cheap redundancy filter:
      - INTERSECTION: each operand must be a STRICT SUPERSET of target (otherwise trivial or impossible)
      - UNION: each operand must be a STRICT SUBSET of target (otherwise trivial or impossible)
    This removes things like: target == A and (A ∩ B) where B is a superset of A.
    """
    if op == "INTERSECTION":
        # For intersection to equal target, target must be subset of each operand.
        # If any operand equals target, then combo is trivial (intersection can't add elements).
        return all(target < s for s in sets)  # strict superset
    elif op == "UNION":
        # For union to equal target, each operand must be subset of target.
        # If any operand equals target, combo is trivial.
        return all(s < target for s in sets)  # strict subset
    else:
        raise ValueError(f"Unknown op: {op}")


def _is_minimal_combo(
    target: Set[str],
    combo: Tuple[str, ...],
    name_to_set: Dict[str, Set[str]],
    op: str,
) -> bool:
    """
    Minimality criterion:
    - combo yields target under op
    - no proper sub-combination of combo yields target

    This removes redundant supersets like:
      (A ∩ B) = target but A alone already equals target
    and also removes bigger combos when a smaller combo already explains target.
    """
    # Any proper subset yields target => not minimal
    for k in range(2, len(combo)):
        for sub in itertools.combinations(combo, k):
            sub_sets = [name_to_set[n] for n in sub]
            if _apply_op(sub_sets, op) == target:
                return False
    return True


def combo_matches_against_target_set(
    target: Set[str],
    name_to_set: Dict[str, Set[str]],
    max_combo: int,
    ops: Tuple[str, ...] = ("UNION", "INTERSECTION"),
) -> List[ComboMatch]:
    """
    Find NON-REDUNDANT combinations of names (size 2..max_combo) whose UNION/INTERSECTION equals target.

    Non-redundant means:
      - Not trivial (no operand already equals target)
      - Not "obvious" redundancy:
          INTERSECTION: every operand must be a strict superset of target
          UNION: every operand must be a strict subset of target
      - Minimal: no smaller sub-combo already yields target
    """
    names = list(name_to_set.keys())
    out: List[ComboMatch] = []

    for op in ops:
        for k in range(2, max_combo + 1):
            for combo in itertools.combinations(names, k):
                sets = [name_to_set[n] for n in combo]

                # Fast discard of trivial/redundant candidates
                if not _is_candidate_nontrivial(target, sets, op):
                    continue

                # Check if combo yields target
                if _apply_op(sets, op) != target:
                    continue

                # Check minimality (no smaller combo yields target)
                if not _is_minimal_combo(target, combo, name_to_set, op):
                    continue

                out.append(ComboMatch(op=op, items=combo))

    return out


def combos_of_mi_that_equal_gt_topics(
        mi_to_docs: Dict[str, Set[str]], topic_to_docs: Dict[str, Set[str]], max_mi_combo: int, ) -> Dict[
    str, List[ComboMatch]]:
    """
    For each GT topic, find Mi-combinations (union/intersection) whose doc set equals that topic.
    Returns:
        dict topic -> list of ComboMatch over Mi names
    """
    topic_to_matches: Dict[str, List[ComboMatch]] = {}
    for topic, t_docs in topic_to_docs.items():
        matches = combo_matches_against_target_set(target=t_docs, name_to_set=mi_to_docs, max_combo=max_mi_combo,
                ops=("UNION", "INTERSECTION"), )
        if matches:
            topic_to_matches[topic] = matches
    return topic_to_matches


# ----------------------------
# Output helpers
# ----------------------------

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(rows: List[Dict[str, str]], path: str, fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ----------------------------
# Main functionality
# ----------------------------
def map_mi_to_gt_topics(
cxt_path: str,
    gt_path: str,
    out_dir: str = "results/context_comparison/MLB_GT",
    max_gt_combo: int = 3,
    max_mi_combo: int = 3,
) -> None:
    """
    Compare FCA attributes (Mi) from a Burmeister context against
    ground-truth (GT) topic labels and store all exact, non-redundant matches.

    This function performs:

    1) Exact matches:
       - Mi == GT topic

    2) Mi explained as minimal, non-redundant combinations
       (UNION / INTERSECTION) of GT topics.

    3) GT topics explained as minimal, non-redundant combinations
       (UNION / INTERSECTION) of Mi extents.

    Redundant explanations such as:
        M4 ∩ M3 = TopicX
    where M4 already equals TopicX and M3 is a superset,
    are automatically filtered out.

    Parameters
    ----------
    cxt_path : str
        Path to Burmeister .cxt file (Mi context).

    gt_path : str
        Path to ground-truth FCA JSON file
        (fca_gt_context.json format).

    out_dir : str, optional
        Output directory for results.
        Default: "results/context_comparison/MLB_GT"

    max_gt_combo : int, optional
        Maximum number of GT topics to combine when explaining a single Mi.
        Default: 3

    max_mi_combo : int, optional
        Maximum number of Mi attributes to combine when explaining a GT topic.
        Default: 3

    Output
    ------
    Writes:
        matches.json
        matches.csv

    Only written if at least one match exists.

    Returns
    -------
    None
    """

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    out_dir = ensure_dir(PROJECT_ROOT / out_dir)

    # ----------------------------
    # Parse inputs
    # ----------------------------
    objects, attributes, rows = read_burmeister_cxt(PROJECT_ROOT / cxt_path)
    mi_to_docs = cxt_attribute_extents(objects, attributes, rows)

    gt_cols, gt_index, gt_data = read_gt_context_json(PROJECT_ROOT / gt_path)
    topic_to_docs = gt_topic_docsets(gt_cols, gt_index, gt_data)

    # Doc universe sanity
    cxt_docs = set(objects)
    gt_docs = set(gt_index)

    # ----------------------------
    # Collect ALL matches into one list
    # ----------------------------
    match_rows: List[Dict[str, str]] = []
    match_json: List[dict] = []

    def add_match(
        direction: str,
        target_name: str,
        target_size: int,
        source_op: str,
        source_items: List[str],
    ) -> None:
        """
        direction:
          - 'Mi_EQUALS_GT'          (single exact)
          - 'Mi_EQUALS_GT_COMBO'    (Mi explained by GT topic combo)
          - 'GT_EQUALS_MI_COMBO'    (GT explained by Mi combo)
        """
        match_rows.append({
            "direction": direction,
            "target": target_name,
            "target_size": str(target_size),
            "op": source_op,
            "items": " + ".join(source_items),
        })
        match_json.append({
            "direction": direction,
            "target": target_name,
            "target_size": target_size,
            "op": source_op,
            "items": source_items,
        })

    # ----------------------------
    # 1) Single Mi == single GT topic
    # ----------------------------
    for mi, mi_docs in sorted(mi_to_docs.items(), key=lambda x: x[0]):
        exact_topics = exact_set_matches(mi_docs, topic_to_docs)
        for t in exact_topics:
            add_match(
                direction="Mi_EQUALS_GT",
                target_name=f"{mi} == {t}",
                target_size=len(mi_docs),
                source_op="EQUALS",
                source_items=[mi, t],
            )

    # ----------------------------
    # 2) Mi explained by combos of GT topics (non-redundant)
    # ----------------------------
    for mi, mi_docs in sorted(mi_to_docs.items(), key=lambda x: x[0]):
        gt_combos = combo_matches_against_target_set(
            target=mi_docs,
            name_to_set=topic_to_docs,
            max_combo=max_gt_combo,
            ops=("UNION", "INTERSECTION"),
        )
        for cm in gt_combos:
            add_match(
                direction="Mi_EQUALS_GT_COMBO",
                target_name=mi,
                target_size=len(mi_docs),
                source_op=cm.op,
                source_items=list(cm.items),  # GT topics
            )

    # ----------------------------
    # 3) GT topic explained by combos of Mi (non-redundant)
    # ----------------------------
    for topic, t_docs in sorted(topic_to_docs.items(), key=lambda x: x[0]):
        mi_combos = combo_matches_against_target_set(
            target=t_docs,
            name_to_set=mi_to_docs,
            max_combo=max_mi_combo,
            ops=("UNION", "INTERSECTION"),
        )
        for cm in mi_combos:
            add_match(
                direction="GT_EQUALS_MI_COMBO",
                target_name=topic,
                target_size=len(t_docs),
                source_op=cm.op,
                source_items=list(cm.items),  # Mi attributes
            )

    # Only save if there is at least one match (as requested)
    payload = {
        "inputs": {"cxt": cxt_path, "gt": gt_path},
        "doc_universe_check": {
            "cxt_num_docs": len(cxt_docs),
            "gt_num_docs": len(gt_docs),
            "missing_in_gt_sample": sorted(list(cxt_docs - gt_docs))[:50],
            "missing_in_cxt_sample": sorted(list(gt_docs - cxt_docs))[:50],
        },
        "params": {"max_gt_combo": max_gt_combo, "max_mi_combo": max_mi_combo},
        "matches": match_json,
    }

    if match_json:
        write_json(payload, os.path.join(out_dir, "matches.json"))
        write_csv(
            match_rows,
            os.path.join(out_dir, "matches.csv"),
            fieldnames=["direction", "target", "target_size", "op", "items"],
        )
        print(f"Done. Wrote {len(match_json)} matches to: {out_dir}")
        print("Key files: matches.json, matches.csv")
    else:
        print("No exact matches found (no output files written).")


if __name__ == "__main__":
    print("Start MLB ground-truth comparison:")
    map_mi_to_gt_topics(cxt_path="resources/banksearch/ground_truth/mlb_expanded.cxt",
            gt_path="resources/banksearch/ground_truth/fca_gt_context.json",
            out_dir = "results/context_comparison/MLB_GT",
    max_gt_combo=3, max_mi_combo=3, )

    print("Start topic model comparison:")
    map_mi_to_gt_topics(cxt_path="resources/banksearch/topic_model/banksearch_0.05_iceberg.cxt",
                        gt_path="resources/banksearch/ground_truth/fca_gt_context.json",
                        out_dir="results/context_comparison/topic_model", max_gt_combo=3, max_mi_combo=3, )