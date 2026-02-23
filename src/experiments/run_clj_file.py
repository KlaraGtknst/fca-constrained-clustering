import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import networkx as nx
from matplotlib import pyplot as plt
from edn_format import loads

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _contains_bool(value):
    if isinstance(value, bool):
        return True
    if isinstance(value, list):
        return any(_contains_bool(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_bool(item) for item in value.values())
    return False


def _convert_bools_to_ints(value):
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, list):
        return [_convert_bools_to_ints(item) for item in value]
    if isinstance(value, dict):
        return {key: _convert_bools_to_ints(item) for key, item in value.items()}
    return value


def ensure_zero_one_json(source_path):
    translated_path = source_path.replace(".json", "_01.json")
    try:
        with open(source_path, "r", encoding="utf-8") as source_file:
            data = json.load(source_file)
    except FileNotFoundError:
        print(f"File {source_path} not found. Current pwd: {os.getcwd()}")
        raise (FileNotFoundError)

    if not _contains_bool(data):
        return source_path

    if os.path.exists(translated_path):
        with open(translated_path, "r", encoding="utf-8") as translated_file:
            translated_data = json.load(translated_file)
        if not _contains_bool(translated_data):
            return translated_path

    converted = _convert_bools_to_ints(data)
    with open(translated_path, "w", encoding="utf-8") as translated_file:
        json.dump(converted, translated_file, indent=2, ensure_ascii=True)
    return translated_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
context_path = ensure_zero_one_json(
        str(PROJECT_ROOT) + "/resources/banksearch/topic_model/fca_topic_model_context.json"
)

min_support = 0.05

cmd = [
    "clojure",
    "-M",
    "-e",
    '(load-file "src/experiments/iceberg_lattice.clj") '
    f'(run-iceberg "{context_path}" {min_support} "resources/banksearch/topic_model/banksearch_{min_support}_iceberg.edn")',
]


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


edn_path = f"resources/banksearch/topic_model/banksearch_{min_support}_iceberg.edn"
svg_path = (
    f"resources/banksearch/topic_model/plots/banksearch_{min_support}_iceberg.svg"
)
iceberg_context_csv_path = "resources/banksearch/topic_model/iceberg_context.csv"
cmd = [
    "clojure",
    "-M",
    "-e",
    '(load-file "src/experiments/iceberg_lattice.clj") '
    f'(run-iceberg "{context_path}" {min_support} "{edn_path}")',
]

try:
    path = "/Users/klara/Developer/fca-constrained-clustering"
    if not os.path.exists(path):
        path = "/Users/klara/Developer/Uni/FCA/fca-constrained-clustering"
    out = subprocess.check_output(
        cmd,
        text=True,
        cwd=path,
        stderr=subprocess.STDOUT,
    )
    logger.info(out)
    if os.path.exists(edn_path):
        Path(svg_path).parent.mkdir(parents=True, exist_ok=True)
        plot_lattice_from_edn(edn_path, svg_path, min_support)
        logger.info(f"Saved iceberg lattice SVG to {svg_path}")
        if os.path.exists(iceberg_context_csv_path):
            logger.info(f"Saved iceberg context CSV to {iceberg_context_csv_path}")

        iceberg_concepts = read_edn_concepts(edn_path)
        logger.info(f"Read {len(iceberg_concepts)} iceberg concepts from {edn_path}")

except subprocess.CalledProcessError as e:
    logger.error(e.output)
