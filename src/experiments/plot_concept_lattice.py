import os
import sys
from pathlib import Path

import networkx as nx
from edn_format import loads
import logging

from matplotlib import pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
min_support = 0.05
def read_edn_concepts(path: str):
    with open(path, "r") as f:
        edn_data = loads(f.read())
    logger.info(f"Reading EDN concepts from {path}...")  # {edn_data}
    return edn_data


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


def plot_lattice_from_edn(edn_path: str, svg_path: str, min_support: float):
    concepts = read_edn_concepts(edn_path)
    for concept in concepts:
        logger.info(
            f"n docs {len(concept[0])} n topics {len(concept[1])}"
        )
    edges = _hasse_edges(concepts)
    graph = nx.DiGraph()
    for idx, (_, intent) in enumerate(concepts):
        graph.add_node(idx, layer=len(intent))
    graph.add_edges_from(edges)
    pos = nx.multipartite_layout(graph, subset_key="layer", align="horizontal")
    # Flip vertically so the top (empty intent) is at the top of the plot.
    # Then rescale to add horizontal spacing and reduce vertical spacing.
    pos = {node: (x, -y) for node, (x, y) in pos.items()}
    plt.figure(figsize=(18, 5))
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
    # Draw two labels per node:
    # top = intent attributes, bottom = size of extent (number of objects).
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
    plt.title(f"Iceberg Concept Lattice\nmin_support={min_support}\nNode labels: top = intent (attributes), bottom = |A| (number of objects).", fontsize=12)
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close()


cxt_path = f"resources/banksearch/topic_model/banksearch_{min_support}_iceberg"
svg_path = (
f"resources/banksearch/topic_model/plots/banksearch_{min_support}_iceberg.svg"
)
iceberg_context_csv_path = "resources/banksearch/topic_model/iceberg_context.csv"
path = "/Users/klara/Developer/fca-constrained-clustering"

edn_path = Path(path) / (cxt_path + ".edn")
print(edn_path)
if os.path.exists(edn_path):
    logger.info("Path exists.")
    Path(svg_path).parent.mkdir(parents=True, exist_ok=True)
    plot_lattice_from_edn(edn_path, svg_path, min_support)
    logger.info(f"Saved iceberg lattice SVG to {svg_path}")

    iceberg_concepts = read_edn_concepts(edn_path)
    logger.info(f"Read {len(iceberg_concepts)} iceberg concepts from {edn_path}")
else:
    logger.error(f"Path {cxt_path} does not exist, current pwd {os.getcwd()}")

