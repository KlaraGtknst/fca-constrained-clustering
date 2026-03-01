#!/usr/bin/env python3
"""
Concept similarity between two FCA contexts (via concept extents).

Given two *concept lists* (e.g., exported as EDN from conexp-clj), this script
computes the pairwise similarity of concepts across contexts based on their
EXTENTS (document sets).

Similarity = Jaccard(A, B) = |A ∩ B| / |A ∪ B|
where A and B are extents (sets of document IDs).

Object universes:
- If the two contexts do not have identical object sets, we restrict extents to
  the intersection of object IDs (common documents) before computing similarity.
  This makes the score comparable when one context is a subset of the other.

Inputs:
- context A concepts EDN: vector of [extent intent]
- context B concepts EDN: vector of [extent intent]

Outputs (written under --out-dir):
- similarity_matrix.csv        (shape: #concepts_A x #concepts_B)
- similarity_matrix.npy        (NumPy binary, same shape)
- similarity_metadata.json     (sizes, overlap stats, parameters)
- heatmap.png                  (matplotlib imshow heatmap; default colormap)
- top_matches.csv              (top-N best matches per concept in A)

Example:
  python concept_similarity.py \
    --a-edn results/context_comparison/mlb_expanded_concepts.edn \
    --b-edn results/context_comparison/other_concepts.edn \
    --out-dir results/context_comparison/CONCEPT_SIM \
    --top-k 5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
from edn_format import loads as edn_loads
import matplotlib.pyplot as plt


# ----------------------------
# Data model
# ----------------------------

@dataclass(frozen=True)
class Concept:
    """
    A Formal Concept represented by:
      - extent: set of object/document IDs
      - intent: set of attribute IDs (kept for metadata/labeling, not used for similarity)
    """
    extent: Set[str]
    intent: Set[str]


# ----------------------------
# IO helpers
# ----------------------------

def ensure_dir(path: str) -> str:
    """Create output directory if needed; return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def read_edn_concepts(path: str) -> List[Concept]:
    """
    Read concepts from an EDN file.

    Expected EDN structure:
      [ [#{...extent...} #{...intent...}]
        [#{...} #{...}]
        ... ]

    Returns:
      List[Concept] where extent/intent are python sets of strings.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = edn_loads(f.read())

    concepts: List[Concept] = []
    for idx, item in enumerate(data):
        if not (isinstance(item, Sequence) and len(item) == 2):
            raise ValueError(f"EDN concept entry {idx} is not [extent intent]: {item!r}")
        extent_raw, intent_raw = item

        # edn_format typically converts EDN sets to Python set; members may be str/Keyword/etc.
        extent = {str(x) for x in extent_raw}
        intent = {str(x) for x in intent_raw}

        concepts.append(Concept(extent=extent, intent=intent))

    return concepts


def write_json(obj: Any, path: str) -> None:
    """Write JSON with indentation."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv_matrix(matrix: np.ndarray, path: str) -> None:
    """
    Save a numeric matrix as CSV (no header).
    For large matrices this can be big; NumPy .npy is also written.
    """
    np.savetxt(path, matrix, delimiter=",", fmt="%.6f")


# ----------------------------
# Similarity computation
# ----------------------------

def jaccard(a: Set[str], b: Set[str]) -> float:
    """
    Compute Jaccard similarity of two sets: |A∩B|/|A∪B|.

    Edge case:
      If both are empty after restriction, return 1.0 (identical emptiness).
    """
    if not a and not b:
        return 1.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return float(inter) / float(union) if union else 1.0


def restrict_to_common_objects(concepts: Sequence[Concept], common: Set[str]) -> List[Concept]:
    """
    Return concepts with extents restricted to the given common object set.
    """
    return [Concept(extent=c.extent.intersection(common), intent=c.intent) for c in concepts]


def concept_similarity_matrix(
    concepts_a: Sequence[Concept],
    concepts_b: Sequence[Concept],
) -> np.ndarray:
    """
    Compute pairwise Jaccard similarities between concepts in A and B
    based on extents.

    Returns:
      matrix shape (len(concepts_a), len(concepts_b))
    """
    n_a = len(concepts_a)
    n_b = len(concepts_b)
    sim = np.zeros((n_a, n_b), dtype=np.float32)

    # Simple nested loops are often fine up to ~10k x 10k is too big.
    # If you expect massive lattices, consider sampling or blocking.
    for i, ca in enumerate(concepts_a):
        ea = ca.extent
        for j, cb in enumerate(concepts_b):
            sim[i, j] = jaccard(ea, cb.extent)

    return sim


# ----------------------------
# Reporting helpers
# ----------------------------

def top_k_matches(sim: np.ndarray, k: int) -> List[Tuple[int, int, float]]:
    """
    For each row i (concept A_i), find the top-k columns j (concept B_j)
    by similarity.

    Returns a flat list of (i, j, sim[i,j]) across all i, limited to top-k per i.
    """
    results: List[Tuple[int, int, float]] = []
    k = max(1, int(k))

    for i in range(sim.shape[0]):
        row = sim[i]
        # argsort descending:
        top_idx = np.argsort(row)[::-1][:k]
        for j in top_idx:
            results.append((i, int(j), float(row[j])))

    return results


def save_top_matches_csv(
    matches: List[Tuple[int, int, float]],
    concepts_a: Sequence[Concept],
    concepts_b: Sequence[Concept],
    path: str,
) -> None:
    """
    Save top matches to CSV with some helpful metadata (extent sizes, intent sizes).
    """
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "a_idx", "b_idx", "similarity",
                "a_extent_size", "b_extent_size",
                "a_intent_size", "b_intent_size",
            ],
        )
        w.writeheader()
        for i, j, s in matches:
            ca = concepts_a[i]
            cb = concepts_b[j]
            w.writerow({
                "a_idx": i,
                "b_idx": j,
                "similarity": f"{s:.6f}",
                "a_extent_size": len(ca.extent),
                "b_extent_size": len(cb.extent),
                "a_intent_size": len(ca.intent),
                "b_intent_size": len(cb.intent),
            })

def save_heatmap(sim: np.ndarray, path: str, title: str,
                 xlabel: str = "", ylabel: str = "") -> None:
    plt.figure()
    plt.imshow(sim, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

def infer_context_label(path: str, default_label: str) -> str:
    """
    Infer semantic context label from file path.

    Rules:
      - If 'mlb' in path (case-insensitive) → 'MLB'
      - If 'iceberg' in path → 'Iceberg Topic Model'
      - Otherwise → default_label
    """
    p = path.lower()
    if "mlb" in p:
        return "MLB"
    if "iceberg" in p:
        return "Iceberg Topic Model"
    return default_label

def json_safe_pairs(high_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert frozenset/set fields in similarity-pair dicts to JSON-serializable
    representations (sorted lists).
    """
    out: List[Dict[str, Any]] = []
    for row in high_pairs:
        out.append({
            **row,
            "a_intent": sorted(row["a_intent"]),
            "b_intent": sorted(row["b_intent"]),
        })
    return out

def find_unique_high_similarity_pairs(
    sim: np.ndarray,
    concepts_a: Sequence[Concept],
    concepts_b: Sequence[Concept],
    threshold: float,
) -> List[Dict[str, Any]]:
    """
    Find maximum non-overlapping concept pairs above threshold.

    Each concept from A and B can appear at most once.
    Greedy selection by highest similarity first.

    Returns:
        List of selected pair dicts.
    """

    candidates: List[Tuple[float, int, int]] = []

    # Collect all candidate pairs above threshold
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            score = float(sim[i, j])
            if score >= threshold:
                candidates.append((score, i, j))

    # Sort descending by similarity
    candidates.sort(reverse=True, key=lambda x: x[0])

    used_a = set()
    used_b = set()
    results = []

    for score, i, j in candidates:

        if i in used_a or j in used_b:
            continue  # already matched

        a = concepts_a[i]
        b = concepts_b[j]

        inter = a.extent & b.extent
        union = a.extent | b.extent

        results.append({
            "similarity": score,
            "a_intent": frozenset(a.intent),
            "b_intent": frozenset(b.intent),
            "intersection_size": len(inter),
            "union_size": len(union),
            "a_extent_size": len(a.extent),
            "b_extent_size": len(b.extent),
        })

        used_a.add(i)
        used_b.add(j)

    return results

# ----------------------------
# Script entry point
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Compute concept-extent similarity between two contexts.")
    ap.add_argument("--a-edn",  default="results/context_comparison/mlb_expanded_concepts.edn", required=False, help="EDN concepts file for context A (vector of [extent intent]).")
    ap.add_argument("--b-edn",  default="resources/banksearch/topic_model/banksearch_0.05_iceberg.edn", required=False, help="EDN concepts file for context B (vector of [extent intent]).")
    ap.add_argument("--out-dir", default="results/context_comparison/CONCEPT_SIM", help="Output directory.")
    ap.add_argument("--top-k", type=int, default=5, help="Top-k matches per concept in A to save.")
    ap.add_argument("--max-a", type=int, default=0, help="If >0, limit to first max-a concepts of A.")
    ap.add_argument("--max-b", type=int, default=0, help="If >0, limit to first max-b concepts of B.")
    ap.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.5,
        help="Only save concept pairs with similarity >= threshold."
    )
    args = ap.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    out_dir = ensure_dir(PROJECT_ROOT / args.out_dir)
    path_a = str(PROJECT_ROOT / args.a_edn)
    path_b = str(PROJECT_ROOT / args.b_edn)

    label_a = infer_context_label(path_a, "Context A")
    label_b = infer_context_label(path_b, "Context B")

    # Load concepts
    concepts_a = read_edn_concepts(PROJECT_ROOT / args.a_edn)
    concepts_b = read_edn_concepts(PROJECT_ROOT / args.b_edn)

    # Optional limiting (useful if lattices are huge)
    if args.max_a and args.max_a > 0:
        concepts_a = concepts_a[: args.max_a]
    if args.max_b and args.max_b > 0:
        concepts_b = concepts_b[: args.max_b]

    # Determine common object universe and restrict extents
    objs_a = set().union(*(c.extent for c in concepts_a)) if concepts_a else set()
    objs_b = set().union(*(c.extent for c in concepts_b)) if concepts_b else set()
    common_objs = objs_a.intersection(objs_b)

    concepts_a_r = restrict_to_common_objects(concepts_a, common_objs)
    concepts_b_r = restrict_to_common_objects(concepts_b, common_objs)

    # Compute similarity matrix
    sim = concept_similarity_matrix(concepts_a_r, concepts_b_r)

    # Save matrix
    write_csv_matrix(sim, os.path.join(out_dir, "similarity_matrix.csv"))
    np.save(os.path.join(out_dir, "similarity_matrix.npy"), sim)

    # Save heatmap
    save_heatmap(
        sim,
        os.path.join(out_dir, "heatmap.svg"),
        title=f"Concept extent similarity (Jaccard)\n{label_a} vs {label_b}",
        xlabel=f"{label_b} Concepts",
        ylabel=f"{label_a} Concepts",
    )

    # -------------------------------------------------
    # Save high similarity pairs above threshold
    # -------------------------------------------------
    if sim.size:
        print(f"Finding concept pairs with similarity >= {args.similarity_threshold}...{concepts_b_r}")
        high_pairs = find_unique_high_similarity_pairs(
            sim,
            concepts_a_r,
            concepts_b_r,
            threshold=args.similarity_threshold,
        )

        if high_pairs:
            # Save JSON
            print(high_pairs)
            write_json(
                json_safe_pairs(high_pairs),
                os.path.join(out_dir, "high_similarity_pairs.json"),
            )

            fieldnames = [
                "similarity",
                "a_intent",
                "b_intent",
                "a_extent_size",
                "b_extent_size",
                "intersection_size",
                "union_size",
            ]

            with open(os.path.join(out_dir, "high_similarity_pairs.csv"), "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for row in high_pairs:
                    writer.writerow({
                        "similarity": f"{row['similarity']:.6f}",
                        # CSV-friendly: stable order
                        "a_intent": " ".join(sorted(row["a_intent"])),
                        "b_intent": " ".join(sorted(row["b_intent"])),
                        "a_extent_size": row["a_extent_size"],
                        "b_extent_size": row["b_extent_size"],
                        "intersection_size": row["intersection_size"],
                        "union_size": row["union_size"],
                    })

            print(
                f"Saved {len(high_pairs)} pairs with similarity >= "
                f"{args.similarity_threshold}"
            )
        else:
            print(
                f"No concept pairs found with similarity >= "
                f"{args.similarity_threshold}"
            )

    # Save top matches (only meaningful if there is at least one concept)
    if sim.size and args.top_k > 0:
        matches = top_k_matches(sim, args.top_k)
        save_top_matches_csv(
            matches,
            concepts_a_r,
            concepts_b_r,
            os.path.join(out_dir, "top_matches.csv"),
        )

    # Metadata for reproducibility/debugging
    meta = {
        "inputs": {
            "a_edn": args.a_edn,
            "b_edn": args.b_edn,
        },
        "counts": {
            "concepts_a": len(concepts_a),
            "concepts_b": len(concepts_b),
            "common_objects": len(common_objs),
        },
        "object_universe": {
            "objects_in_a_union_of_extents": len(objs_a),
            "objects_in_b_union_of_extents": len(objs_b),
            "restricted_to_common_objects": True,
        },
        "params": {
            "top_k": args.top_k,
            "max_a": args.max_a,
            "max_b": args.max_b,
        },
        "outputs": {
            "similarity_matrix_csv": "similarity_matrix.csv",
            "similarity_matrix_npy": "similarity_matrix.npy",
            "heatmap": "heatmap.png",
            "top_matches_csv": "top_matches.csv" if sim.size and args.top_k > 0 else None,
        },
    }
    write_json(meta, os.path.join(out_dir, "similarity_metadata.json"))

    print(f"Saved results to: {out_dir}")
    print("Files:")
    print(" - similarity_matrix.csv")
    print(" - similarity_matrix.npy")
    print(" - similarity_metadata.json")
    print(" - heatmap.png")
    print(" - high_similarity_pairs.csv")
    if sim.size and args.top_k > 0:
        print(" - top_matches.csv")


if __name__ == "__main__":
    main()