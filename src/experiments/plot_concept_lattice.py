from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple, Optional

import networkx as nx
from edn_format import loads
from matplotlib import pyplot as plt


Concept = Tuple[Sequence[Any], Sequence[Any]]  # (extent, intent)


class IcebergLatticePlotter:
    """
    Reads iceberg concepts from an EDN file and plots the concept lattice (Hasse diagram) to SVG.

    Public API:
      - plot(edn_path, svg_path, min_support=..., figsize=...)
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def plot(
        self,
        edn_path: str | Path,
        svg_path: str | Path,
        *,
        min_support: Optional[float] = None,
        figsize: tuple[float, float] = (18, 5),
        node_size: int = 20,
        edge_width: float = 0.6,
        font_size: int = 7,
        omit_transitive_intents: bool = False,
    ) -> None:
        """
        Read concepts from `edn_path` and save an SVG plot to `svg_path`.

        If omit_transitive_intents=True, each node label shows only those intent
        attributes that do NOT already occur in any "above" concept (i.e., any concept
        with a proper-subset intent).
        """
        edn_path = Path(edn_path)
        svg_path = Path(svg_path)

        if not edn_path.exists():
            raise FileNotFoundError(f"EDN path does not exist: {edn_path}")

        concepts = self._read_edn_concepts(edn_path)
        edges = self._hasse_edges(concepts)
        graph, pos = self._build_graph_and_layout(concepts, edges)

        svg_path.parent.mkdir(parents=True, exist_ok=True)
        self._draw_and_save(
            graph=graph,
            pos=pos,
            concepts=concepts,
            svg_path=svg_path,
            min_support=min_support,
            figsize=figsize,
            node_size=node_size,
            edge_width=edge_width,
            font_size=font_size,
            omit_transitive_intents=omit_transitive_intents,  # NEW
        )

        self.logger.info("Saved iceberg lattice SVG to %s", svg_path)

    # -------------------------
    # Internals
    # -------------------------

    def _read_edn_concepts(self, path: Path) -> List[Concept]:
        self.logger.info("Reading EDN concepts from %s...", path)
        data = loads(path.read_text(encoding="utf-8"))
        # Expecting: [(extent, intent), ...]
        return list(data)

    def _hasse_edges(self, concepts: Sequence[Concept]) -> List[tuple[int, int]]:
        extents = [set(extent) for extent, _ in concepts]
        sizes = [len(extent) for extent in extents]
        order = sorted(range(len(extents)), key=lambda i: sizes[i])

        edges: List[tuple[int, int]] = []
        for i in order:
            sup_candidates: List[int] = []
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

    def _build_graph_and_layout(
        self,
        concepts: Sequence[Concept],
        edges: Sequence[tuple[int, int]],
    ) -> tuple[nx.DiGraph, dict[int, tuple[float, float]]]:
        graph = nx.DiGraph()
        for idx, (_, intent) in enumerate(concepts):
            graph.add_node(idx, layer=len(intent))
        graph.add_edges_from(edges)

        pos = nx.multipartite_layout(graph, subset_key="layer", align="horizontal")
        # Flip vertically so empty intent is at the top
        pos = {node: (x, -y) for node, (x, y) in pos.items()}
        return graph, pos

    def _non_transitive_intent_labels(self, concepts: Sequence[Concept]) -> List[List[str]]:
        """
        For each concept i, return the list of intent attribute labels that are NOT present
        in any concept 'above' it, where 'above' is defined as any concept j whose intent
        is a proper subset of intent(i).
        """
        intent_sets = [set(map(str, intent)) for _, intent in concepts]

        # Process from small intents to large intents (top -> bottom)
        order = sorted(range(len(concepts)), key=lambda i: (len(intent_sets[i]), i))

        above_union: List[set[str]] = [set() for _ in range(len(concepts))]
        for i in order:
            Ii = intent_sets[i]
            if not Ii:
                above_union[i] = set()
                continue

            u: set[str] = set()
            # any proper subset intent counts as "above"
            for j in order:
                if j == i:
                    continue
                Ij = intent_sets[j]
                if len(Ij) >= len(Ii):
                    break  # because order is nondecreasing by size
                if Ij and Ij.issubset(Ii):
                    u |= Ij
            above_union[i] = u

        filtered: List[List[str]] = []
        for i, Ii in enumerate(intent_sets):
            shown = sorted(Ii - above_union[i])
            filtered.append(shown)
        return filtered

    def _draw_and_save(
        self,
        *,
        graph: nx.DiGraph,
        pos: dict[int, tuple[float, float]],
        concepts: Sequence[Concept],
        svg_path: Path,
        min_support: Optional[float],
        figsize: tuple[float, float],
        node_size: int,
        edge_width: float,
        font_size: int,
        omit_transitive_intents: bool,  # NEW
    ) -> None:
        plt.figure(figsize=figsize)

        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        y_range = (max(ys) - min(ys)) if ys else 1.0
        y_label_offset = 0.02 * y_range
        x_label_offset = 0.01

        nx.draw_networkx(
            graph,
            pos=pos,
            with_labels=False,
            node_size=node_size,
            width=edge_width,
            arrows=False,
        )

        filtered_intents: Optional[List[List[str]]] = None
        if omit_transitive_intents:
            filtered_intents = self._non_transitive_intent_labels(concepts)

        for idx, (extent, intent) in enumerate(concepts):
            if omit_transitive_intents and filtered_intents is not None:
                shown_attrs = filtered_intents[idx]
                intent_labels = ",".join(shown_attrs) if shown_attrs else "{}"
            else:
                intent_labels = ",".join(sorted(map(str, intent))) if intent else "{}"

            extent_size = len(extent)
            x, y = pos[idx]

            plt.text(
                x + x_label_offset,
                y + y_label_offset,
                intent_labels,
                ha="center",
                va="bottom",
                fontsize=font_size,
            )
            plt.text(
                x + x_label_offset,
                y - y_label_offset,
                f"|A|={extent_size}",
                ha="center",
                va="top",
                fontsize=font_size,
            )

        plt.axis("off")
        title = "Iceberg Concept Lattice"
        if min_support is not None:
            title += f"\nmin_support={min_support}"
        if omit_transitive_intents:
            title += "\n(intent labels omit attributes already occurring in any above concept)"
        title += "\nNode labels: top = intent (attributes), bottom = |A| (number of objects)."
        plt.title(title, fontsize=12)
        plt.savefig(svg_path, format="svg", bbox_inches="tight")
        plt.close()