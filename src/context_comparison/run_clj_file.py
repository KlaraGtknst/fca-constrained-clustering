import json
import logging
import os
from pathlib import Path
import subprocess
from typing import Dict, Any

import networkx as nx
from matplotlib import pyplot as plt
from edn_format import loads


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "context_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def call_clojure_analyze(cxt_path: str, min_support: float = 0.0) -> Dict[str, Any]:
    clj = [
        "clojure",
        "-M",
        "-e",
        ' (load-file "src/context_comparison/context_comparison.clj")'
        f' (clojure.core/println (user/analyze-context "{cxt_path}" {min_support}))',
    ]
    cwd_candidates = [
        str(PROJECT_ROOT),
        str(Path.home() / "Developer" / "Uni" / "FCA" / "fca-constrained-clustering"),
    ]
    cwd = next((p for p in cwd_candidates if os.path.exists(p)), str(PROJECT_ROOT))
    logger.info(f"Calling Clojure analyze on {cxt_path} (cwd={cwd})...")
    out = subprocess.check_output(clj, text=True, cwd=cwd, stderr=subprocess.STDOUT)
    logger.info(out)
    # The function already writes JSON stats; optionally parse last JSON-looking line if needed.
    return {}


def read_edn_concepts(path: str):
    with open(path, "r") as f:
        return loads(f.read())


def _hasse_edges(concepts):
    extents = [set(extent) for extent, _ in concepts]
    sizes = [len(extent) for extent in extents]
    order = sorted(range(len(extents)), key=lambda i: sizes[i])
    edges = []
    for i in order:
        sup_candidates = []
        for j in order:
            if sizes[j] <= sizes[i]:
                continue
            if not extents[i].issubset(extents[j]):
                continue
            is_shadowed = False
            for k in list(sup_candidates):
                if extents[k].issubset(extents[j]):
                    is_shadowed = True
                    break
                if extents[j].issubset(extents[k]):
                    sup_candidates.remove(k)
            if not is_shadowed:
                sup_candidates.append(j)
        edges.extend((i, j) for j in sup_candidates)
    return edges


def plot_lattice_from_edn(edn_path: str, png_path: str, title: str):
    concepts = read_edn_concepts(edn_path)
    edges = _hasse_edges(concepts)
    graph = nx.DiGraph()
    for idx, (_, intent) in enumerate(concepts):
        graph.add_node(idx, layer=len(intent))
    graph.add_edges_from(edges)
    pos = nx.multipartite_layout(graph, subset_key="layer", align="horizontal")
    pos = {node: (x, -y) for node, (x, y) in pos.items()}

    plt.figure(figsize=(20, 6))
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    y_range = (max(ys) - min(ys)) if ys else 1.0
    y_label_offset = 0.02 * y_range
    x_label_offset = 0.01

    nx.draw_networkx(
        graph,
        pos=pos,
        with_labels=False,
        node_size=20,
        width=0.6,
        arrows=False,
    )

    for idx, (extent, intent) in enumerate(concepts):
        intent_labels = ",".join(sorted(map(str, intent))) if intent else "{}"
        extent_size = len(extent)
        x, y = pos[idx]
        plt.text(
            x + x_label_offset,
            y + y_label_offset,
            intent_labels,
            ha="center",
            va="bottom",
            fontsize=7,
        )
        plt.text(
            x + x_label_offset,
            y - y_label_offset,
            f"|A|={extent_size}",
            ha="center",
            va="top",
            fontsize=7,
        )

    plt.axis("off")
    plt.title(title, fontsize=12)
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, format="png", dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Paths provided in the request
    cxt1 = str(PROJECT_ROOT / "resources" / "banksearch" / "topic_model" / "iceberg_context.cxt")
    cxt2 = str(PROJECT_ROOT / "resources" / "banksearch" / "ground_truth" / "mlb_expanded.cxt")

    # Analyze both (full lattice by default)
    call_clojure_analyze(cxt1, 0.0)
    call_clojure_analyze(cxt2, 0.0)

    # Plot PNGs from the EDN concepts that Clojure produced
    # FIXME: use Burmeister format to save formal (iceberg) context
    edn1 = RESULTS_DIR / "iceberg_context_concepts.edn"
    edn2 = RESULTS_DIR / "mlb_expanded_concepts.edn"

    png1 = RESULTS_DIR / "iceberg_context_lattice.png"
    png2 = RESULTS_DIR / "mlb_expanded_lattice.png"

    if edn1.exists():
        plot_lattice_from_edn(str(edn1), str(png1), title="Concept Lattice: iceberg_context.cxt")
        logger.info(f"Saved PNG to {png1}")
    else:
        logger.warning(f"EDN not found: {edn1}")

    if edn2.exists():
        plot_lattice_from_edn(str(edn2), str(png2), title="Concept Lattice: mlb_expanded.cxt")
        logger.info(f"Saved PNG to {png2}")
    else:
        logger.warning(f"EDN not found: {edn2}")

    # Optionally, aggregate stats that the Clojure side already saved
    stats1_path = RESULTS_DIR / "iceberg_context_stats.json"
    stats2_path = RESULTS_DIR / "mlb_expanded_stats.json"

    comparison_path = RESULTS_DIR / "comparison_summary.json"
    summary: Dict[str, Any] = {}
    if stats1_path.exists():
        with open(stats1_path) as f:
            summary["iceberg_context"] = json.load(f)
    if stats2_path.exists():
        with open(stats2_path) as f:
            summary["mlb_expanded"] = json.load(f)

    if summary:
        with open(comparison_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Wrote comparison summary to {comparison_path}")
    else:
        logger.warning("No stats JSON files found to summarize.")
