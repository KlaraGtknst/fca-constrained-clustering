"""
Concept similarity between two FCA contexts (via concept extents).

This script loads two EDN files containing formal concepts in the format:
    [ [#{extent} #{intent}] ... ]

It computes pairwise concept similarity across contexts using Jaccard similarity
on extents (document sets):

    sim(A, B) = |extent(A) ∩ extent(B)| / |extent(A) ∪ extent(B)|

If the two contexts contain different object universes, extents are restricted
to the intersection of object IDs before similarity is computed, so similarities
remain comparable even if one context is a subset of the other.

Outputs (written to --out-dir):
- similarity_matrix.csv / similarity_matrix.npy
- heatmap.svg
- top_matches.csv                  (top-k matches per concept in context A)
- high_similarity_pairs.csv/json   (greedy one-to-one pairs above threshold)
- similarity_metadata.json

Naming:
If 'mlb' appears in a path, that context is labeled "MLB".
If 'iceberg' appears in a path, that context is labeled "Iceberg Topic Model".
Otherwise, fallback labels are used.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from edn_format import loads as edn_loads


# ============================================================
# Data model
# ============================================================

@dataclass(frozen=True)
class Concept:
    """
    Formal concept representation.

    Attributes
    ----------
    extent:
        Set of object/document identifiers.
    intent:
        Set of attribute identifiers. Used as a concept identifier in outputs
        (not used for similarity).
    """
    extent: Set[str]
    intent: Set[str]


# ============================================================
# Utility / IO
# ============================================================

def ensure_dir(path: Path) -> Path:
    """Create output directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(obj: Any, path: Path) -> None:
    """
    Write JSON with indentation.

    Supports sets/frozensets by converting them to sorted lists.
    """
    def _default(o: Any):
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_default)


def write_csv_matrix(matrix: np.ndarray, path: Path) -> None:
    """Save a numeric matrix as CSV (no header)."""
    np.savetxt(path, matrix, delimiter=",", fmt="%.6f")


def infer_context_label(path: str, default_label: str) -> str:
    """
    Infer a semantic context label from a file path.

    Rules (case-insensitive):
      - 'mlb'     -> 'MLB'
      - 'iceberg' -> 'Iceberg Topic Model'
      - else      -> default_label
    """
    p = path.lower()
    if "mlb" in p:
        return "MLB"
    if "iceberg" in p:
        return "Iceberg Topic Model"
    return default_label


def read_edn_concepts(path: Path) -> List[Concept]:
    """
    Read concepts from an EDN file.

    Expected EDN structure:
        [ [#{...extent...} #{...intent...}]
          [#{...} #{...}]
          ... ]

    Returns
    -------
    list[Concept]
        Extent/intent items are converted to Python sets of strings.
    """
    data = edn_loads(path.read_text(encoding="utf-8"))

    concepts: List[Concept] = []
    for idx, item in enumerate(data):
        # edn_format returns its own sequence type; accept anything sequence-like with length 2
        if not (hasattr(item, "__len__") and hasattr(item, "__getitem__") and len(item) == 2):
            raise ValueError(f"EDN concept entry {idx} is not [extent intent]: {item!r}")

        extent_raw, intent_raw = item
        extent = {str(x) for x in extent_raw}
        intent = {str(x) for x in intent_raw}
        concepts.append(Concept(extent=extent, intent=intent))

    return concepts


# ============================================================
# Similarity computation
# ============================================================

def jaccard(a: Set[str], b: Set[str]) -> float:
    """
    Compute Jaccard similarity of two sets: |A∩B| / |A∪B|.

    If both sets are empty (after restriction), return 1.0.
    """
    if not a and not b:
        return 1.0
    union = a | b
    return (len(a & b) / len(union)) if union else 1.0


def restrict_to_common_objects(concepts: Sequence[Concept], common: Set[str]) -> List[Concept]:
    """Restrict all concept extents to the given common object universe."""
    return [Concept(extent=c.extent & common, intent=c.intent) for c in concepts]


def concept_similarity_matrix(concepts_a: Sequence[Concept], concepts_b: Sequence[Concept]) -> np.ndarray:
    """
    Compute pairwise Jaccard similarity matrix across two concept lists.

    Returns
    -------
    np.ndarray
        Shape (len(concepts_a), len(concepts_b))
    """
    print(f"Computing similarity matrix for {len(concepts_a)} concepts in A and {len(concepts_b)} concepts in B...")
    sim = np.zeros((len(concepts_a), len(concepts_b)), dtype=np.float32)
    for i, ca in enumerate(concepts_a):
        ea = ca.extent
        for j, cb in enumerate(concepts_b):
            sim[i, j] = jaccard(ea, cb.extent)
    return sim


# ============================================================
# Reporting / selection
# ============================================================

def save_heatmap(sim: np.ndarray, path: Path, title: str, xlabel: str, ylabel: str) -> None:
    """Save a heatmap image of the similarity matrix."""
    plt.figure()
    plt.imshow(sim, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {path}")


def top_k_matches(sim: np.ndarray, k: int) -> List[Tuple[int, int, float]]:
    """
    For each concept in A (row i), return its top-k matches in B (columns j)
    by descending similarity.

    Returns
    -------
    list[(i, j, score)]
    """
    k = max(1, int(k))
    out: List[Tuple[int, int, float]] = []
    for i in range(sim.shape[0]):
        row = sim[i]
        top_idx = np.argsort(row)[::-1][:k]
        out.extend((i, int(j), float(row[j])) for j in top_idx)
    return out


def save_top_matches_csv(matches: List[Tuple[int, int, float]],
                         concepts_a: Sequence[Concept],
                         concepts_b: Sequence[Concept],
                         path: Path) -> None:
    """Save top-k matches to CSV (indices + extent/intent sizes)."""
    with path.open("w", encoding="utf-8", newline="") as f:
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
            w.writerow({
                "a_idx": i,
                "b_idx": j,
                "similarity": f"{s:.6f}",
                "a_extent_size": len(concepts_a[i].extent),
                "b_extent_size": len(concepts_b[j].extent),
                "a_intent_size": len(concepts_a[i].intent),
                "b_intent_size": len(concepts_b[j].intent),
            })


def find_unique_high_similarity_pairs(sim: np.ndarray,
                                      concepts_a: Sequence[Concept],
                                      concepts_b: Sequence[Concept],
                                      threshold: float) -> List[Dict[str, Any]]:
    """
    Select high-similarity concept pairs under a one-to-one constraint.

    Each concept from A and B can appear at most once. We use a greedy strategy:
      1) collect all pairs with sim >= threshold
      2) sort candidates by similarity descending
      3) pick a pair if neither side has been used

    This is an approximation of maximum-weight bipartite matching, but is
    typically sufficient for exploratory analysis.

    Returns
    -------
    list[dict]
        Each dict includes the concept *intents* as identifiers and overlap stats.
    """
    candidates: List[Tuple[float, int, int]] = [
        (float(sim[i, j]), i, j)
        for i in range(sim.shape[0])
        for j in range(sim.shape[1])
        if float(sim[i, j]) >= threshold
    ]
    candidates.sort(key=lambda x: x[0], reverse=True)

    used_a: Set[int] = set()
    used_b: Set[int] = set()
    results: List[Dict[str, Any]] = []

    for score, i, j in candidates:
        if i in used_a or j in used_b:
            continue

        a = concepts_a[i]
        b = concepts_b[j]
        if len(a.intent) == 0 or len(b.intent) == 0:
            # top element
            continue
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


def save_high_pairs(out_dir: Path, high_pairs: List[Dict[str, Any]]) -> None:
    """
    Save high similarity pairs to JSON + CSV.

    Notes
    -----
    - JSON uses write_json() which converts frozenset -> sorted list.
    - CSV stores intents as sorted space-joined strings.
    """
    write_json(high_pairs, out_dir / "high_similarity_pairs.json")

    fieldnames = [
        "similarity",
        "a_intent",
        "b_intent",
        "a_extent_size",
        "b_extent_size",
        "intersection_size",
        "union_size",
    ]

    with (out_dir / "high_similarity_pairs.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in high_pairs:
            writer.writerow({
                "similarity": f"{row['similarity']:.6f}",
                "a_intent": " ".join(sorted(row["a_intent"])),
                "b_intent": " ".join(sorted(row["b_intent"])),
                "a_extent_size": row["a_extent_size"],
                "b_extent_size": row["b_extent_size"],
                "intersection_size": row["intersection_size"],
                "union_size": row["union_size"],
            })

# ============================================================
# Coherence context export (Burmeister format)
# ============================================================

def read_high_pairs_csv(path: Path) -> List[Dict[str, Any]]:
    """
    Read high similarity pairs from CSV written by save_high_pairs().

    Returns
    -------
    list[dict]
        Each dict has: similarity (float), a_intent (set[str]), b_intent (set[str])
    """
    pairs: List[Dict[str, Any]] = []
    if not path.exists():
        return pairs

    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            # Intents are stored as space-separated strings
            a_intent = set(filter(None, (row["a_intent"] or "").split()))
            b_intent = set(filter(None, (row["b_intent"] or "").split()))
            pairs.append({
                "similarity": float(row["similarity"]),
                "a_intent": a_intent,
                "b_intent": b_intent,
            })
    return pairs


def write_burmeister_context(
    objects: Sequence[str],
    attributes: Sequence[str],
    has_attr: Sequence[Set[str]],
    path: Path,
) -> None:
    """
    Write a formal context in Burmeister format.

    Format:
      B
      <blank>
      <#objects>
      <#attributes>
      <blank>
      <object names...>
      <attribute names...>
      <incidence matrix rows...>   (X or .)

    Parameters
    ----------
    objects:
        Object names (documents).
    attributes:
        Attribute names (here: high-similarity pair identifiers).
    has_attr:
        List of sets; has_attr[j] is the set of objects that have attribute j.
    """
    if len(attributes) != len(has_attr):
        raise ValueError("attributes and has_attr must have the same length")

    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("B\n\n")
        f.write(f"{len(objects)}\n")
        f.write(f"{len(attributes)}\n\n")

        for o in objects:
            f.write(f"{o}\n")
        for a in attributes:
            f.write(f"{a}\n")

        # incidence: one row per object, one column per attribute
        for o in objects:
            row = "".join("X" if o in has_attr[j] else "." for j in range(len(attributes)))
            f.write(row + "\n")

    print(f"Wrote Burmeister context with {len(objects)} objects and {len(attributes)} attributes to {path}")


def build_coherence_context_from_high_pairs(
    concepts_a: Sequence[Concept],
    concepts_b: Sequence[Concept],
    all_objects: Sequence[str],
    high_pairs: Sequence[Dict[str, Any]],
    out_path: Path,
) -> None:
    """
    Build a Burmeister context where:
      - Objects = all documents (all_objects)
      - Attributes = high-similarity concept pairs
      - Incidence: document gets 'X' if it lies in extent(A) ∩ extent(B)

    Parameters
    ----------
    concepts_a, concepts_b:
        Concept lists used to compute similarity (restricted versions).
    all_objects:
        Complete object universe (e.g. sorted(common_objs)).
    high_pairs:
        Output of find_unique_high_similarity_pairs().
    out_path:
        Output file path for Burmeister context.
    """
    # Map intent -> extent (fast lookup)
    a_by_intent = {frozenset(c.intent): c.extent for c in concepts_a}
    b_by_intent = {frozenset(c.intent): c.extent for c in concepts_b}

    attributes: List[str] = []
    has_attr: List[Set[str]] = []

    for idx, row in enumerate(high_pairs):
        a_int = frozenset(row["a_intent"])
        b_int = frozenset(row["b_intent"])

        ea = a_by_intent.get(a_int)
        eb = b_by_intent.get(b_int)
        if ea is None or eb is None:
            continue

        intersection = ea & eb
        # sim = float(row["similarity"])

        attributes.append(f"c{idx}")# f"pair_{idx:04d}_sim_{sim:.3f}"
        has_attr.append(intersection)

    write_burmeister_context(objects=list(all_objects), attributes=attributes, has_attr=has_attr, path=out_path, )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line args."""
    min_supp = 0.05
    ap = argparse.ArgumentParser(description="Compute concept-extent similarity between two contexts.")
    ap.add_argument("--a-edn", default="results/context_comparison/mlb_expanded_concepts.edn")
    ap.add_argument("--b-edn", default=f"resources/banksearch/topic_model/banksearch_{min_supp}_iceberg.edn")
    ap.add_argument("--out-dir", default="results/context_comparison/CONCEPT_SIM")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-a", type=int, default=0, help="If >0, limit to first max-a concepts of A.")
    ap.add_argument("--max-b", type=int, default=0, help="If >0, limit to first max-b concepts of B.")
    ap.add_argument("--similarity-threshold", type=float, default=0.5)
    return ap.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    project_root = Path(__file__).resolve().parents[2]
    a_path = project_root / args.a_edn
    b_path = project_root / args.b_edn
    out_dir = ensure_dir(project_root / args.out_dir)

    label_a = infer_context_label(str(a_path), "Context A")
    label_b = infer_context_label(str(b_path), "Context B")

    # ----------------------------
    # Load concepts
    # ----------------------------
    concepts_a = read_edn_concepts(a_path)
    concepts_b = read_edn_concepts(b_path)

    # Optional limiting (useful if lattices are huge)
    if args.max_a > 0:
        concepts_a = concepts_a[:args.max_a]
    if args.max_b > 0:
        concepts_b = concepts_b[:args.max_b]

    # ----------------------------
    # Restrict to common object universe
    # ----------------------------
    objs_a = set().union(*(c.extent for c in concepts_a)) if concepts_a else set()
    objs_b = set().union(*(c.extent for c in concepts_b)) if concepts_b else set()
    common_objs = objs_a & objs_b

    concepts_a_r = restrict_to_common_objects(concepts_a, common_objs)
    concepts_b_r = restrict_to_common_objects(concepts_b, common_objs)

    # ----------------------------
    # Compute similarity
    # ----------------------------
    sim = concept_similarity_matrix(concepts_a_r, concepts_b_r)

    # ----------------------------
    # Save similarity artifacts
    # ----------------------------
    write_csv_matrix(sim, out_dir / "similarity_matrix.csv")
    np.save(out_dir / "similarity_matrix.npy", sim)

    save_heatmap(
        sim,
        out_dir / "heatmap.svg",
        title=f"Concept extent similarity (Jaccard)\n{label_a} vs {label_b}",
        xlabel=f"{label_b} concepts",
        ylabel=f"{label_a} concepts",
    )

    # ----------------------------
    # Save unique high-similarity pairs (one-to-one)
    # ----------------------------
    if sim.size:
        high_pairs = find_unique_high_similarity_pairs(
            sim, concepts_a_r, concepts_b_r, threshold=args.similarity_threshold
        )
        if high_pairs:
            save_high_pairs(out_dir, high_pairs)

            # --------------------------------------------
            # Build "coherence" context from high pairs CSV
            # --------------------------------------------
            coherence_path = out_dir / "coherence_context.cxt"
            build_coherence_context_from_high_pairs(concepts_a=concepts_a_r, concepts_b=concepts_b_r,
                    all_objects=sorted(common_objs),  # "all documents as objects" (common universe)
                    high_pairs=high_pairs, out_path=coherence_path, )

    # ----------------------------
    # Save top-k matches per concept in A
    # ----------------------------
    if sim.size and args.top_k > 0:
        matches = top_k_matches(sim, args.top_k)
        save_top_matches_csv(matches, concepts_a_r, concepts_b_r, out_dir / "top_matches.csv")

    # ----------------------------
    # Metadata (keep it consistent with actual outputs)
    # ----------------------------
    meta = {
        "contexts": {
            "a_edn": args.a_edn,
            "b_edn": args.b_edn,
            "a_label": label_a,
            "b_label": label_b,
        },
        "counts": {
            "concepts_a": len(concepts_a),
            "concepts_b": len(concepts_b),
            "common_objects": len(common_objs),
        },
        "params": {
            "top_k": args.top_k,
            "max_a": args.max_a,
            "max_b": args.max_b,
            "similarity_threshold": args.similarity_threshold,
        },
        "outputs": {
            "similarity_matrix_csv": "similarity_matrix.csv",
            "similarity_matrix_npy": "similarity_matrix.npy",
            "heatmap": "heatmap.svg",
            "top_matches_csv": "top_matches.csv" if sim.size and args.top_k > 0 else None,
            "high_similarity_pairs_csv": "high_similarity_pairs.csv" if sim.size else None,
            "high_similarity_pairs_json": "high_similarity_pairs.json" if sim.size else None,
            "coherence_context": "coherence_context.cxt" if sim.size else None,
        },
    }
    write_json(meta, out_dir / "similarity_metadata.json")

    print(f"Saved results to: {out_dir}")


if __name__ == "__main__":
    main()