from abc import ABC
from collections import defaultdict
import os
from pathlib import Path
from typing import List, Union
import json
from itertools import combinations
import logging

import edn_format

from attribute_exploration.domain_expert import DomainExpert

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    def __init__(self, dataset_name: str, min_num_children: int = 1):
        """
        Initialize the extractor with a dataset identifier.

        Input:
          `dataset_name`: short dataset key used in filenames, e.g. "banksearch".

        Output:
          Sets `self.dataset_name` and `self.min_num_children`.
        """
        self.dataset_name = dataset_name

        # empty children are bottom nodes, i.e., most specific concepts (topics) without sub-concepts
        # intuitively, they cannot be parents of must-link constraints, because they have no specifications (children/ sub-concepts)

        # FIXME: How to work with one-element children? I suggest a: {b} -> (b,a,a) constraint which enforces b to be clustered with a
        self.min_num_children = min_num_children # minimum number of children to form pairwise constraints
        assert self.min_num_children >= 1, "min_num_children must be at least 1."

    def extract_all_mlb_constraints(self, out_path: Path):
        """
        Extract MLB constraints and write them to disk.

        Input:
          `out_path`: directory to write output files to.

        Output:
          Implementations write one or more files under `out_path`.
        """
        raise NotImplementedError("Subclasses must implement extract_all_mlb_constraints.")

    @staticmethod
    def _node_to_label(node: Union[str, frozenset]) -> str:
        """Convert hierarchy node identifier to a comma-free label string."""
        if isinstance(node, str):
            return node
        assert isinstance(node, frozenset), f"node must be a string or frozenset, but got {type(node)}."
        assert all(isinstance(item, str) for item in node), "frozenset node items must be strings."
        # Keep labels comma-free because constraints are serialized as CSV-like triples.
        return "|".join(sorted(node))

    def _get_mlb_constraints(
        self,
        third_node: Union[str, frozenset],
        siblings: Union[List[Union[str, frozenset]], set[Union[str, frozenset]], tuple[Union[str, frozenset], ...]],
    ) -> List[str]:
        """
        Build MLB constraints for one sibling group against one third node.

        Input:
          `third_node`: label/ID of the comparison node (typically an uncle), e.g. "Science" or "C3".
          `siblings`: iterable of sibling labels/IDs under the same parent, e.g.
          ["Checking", "Loans", "Investments"] or {"C4", "C5"}.

        Output:
          List of MLB constraint strings in "sibling_a,sibling_b,third_node" format.
        """
        assert isinstance(third_node, (str, frozenset)), f"third_node must be a string/frozenset label, but got {type(third_node)}."
        assert isinstance(siblings, (list, set, tuple)), f"siblings must be a list/set/tuple of labels, but got {type(siblings)}."
        assert all(isinstance(node, (str, frozenset)) for node in siblings), (
            f"each sibling must be a string/frozenset label, but got {[type(node) for node in siblings]}"
        )

        min_siblings = max(2, self.min_num_children)
        if len(siblings) < min_siblings:
            return []

        third_label = self._node_to_label(third_node)
        sibling_labels = [self._node_to_label(node) for node in siblings]
        assert "," not in third_label, "third_node label must not contain commas."
        assert all("," not in label for label in sibling_labels), "sibling labels must not contain commas."

        sibling_labels = sorted(set(sibling_labels))
        return [f"{sibling_a},{sibling_b},{third_label}" for sibling_a, sibling_b in combinations(sibling_labels, 2)]

    def _get_constraints_from_hierarchy_dict(self, hierarchy_dict: dict) -> List[str]:
        """
        Build MLB constraints from a hierarchy dictionary using extended sibling-uncle logic.

        Input:
          `hierarchy_dict`: dict mapping parent -> direct children.
          Each parent is treated as sibling of every other parent (implicit root),
          so for:
            parent: {child_a, child_b}
            uncle:  {cousin_a, cousin_b}
          constraints include:
            (child_a, child_b, uncle), (cousin_a, cousin_b, parent)

        Output:
          Flat, sorted list of "sibling_a,sibling_b,uncle" constraint strings.
        """
        assert isinstance(hierarchy_dict, dict), "hierarchy_dict must be a dict."
        assert len(hierarchy_dict) > 0, "hierarchy_dict must not be empty."

        for node, children in hierarchy_dict.items():
            assert isinstance(node, (str, frozenset)), (
                f"hierarchy_dict keys must be strings or frozensets, but got {type(node)}."
            )
            assert isinstance(children, (list, set, tuple)), (
                f"hierarchy_dict values must be iterables, but got {type(children)}."
            )

        nodes = list(hierarchy_dict.keys())
        constraints_set = set()
        # only work on flat hierarchy
        for parent in nodes:
            siblings = hierarchy_dict[parent]
            for uncle in nodes:
                if uncle == parent:
                    continue
                constraints_set.update(
                    self._get_mlb_constraints(third_node=uncle, siblings=siblings)
                )

        return sorted(constraints_set)


class BankSearchGroundTruthExtractor(BaseExtractor):

    def __init__(self):
        """
        Initialize the ground-truth extractor for the BankSearch dataset.

        Input:
          None.

        Output:
          Sets `self.path2category_hierarchy` to:
          "resources/banksearch/ground_truth/category_hierarchy.json".

        Expected input file structure:
          A JSON object mapping parent label -> list of child labels, e.g.
          {
            "Accounts": ["Checking", "Savings"],
            "Loans": ["Mortgage", "Personal"]
          }
        """
        super().__init__(dataset_name="banksearch")
        self.path2category_hierarchy = PROJECT_ROOT / Path(
            "resources/banksearch/ground_truth/category_hierarchy.json"
        )
        assert (
            self.path2category_hierarchy.exists()
            and self.path2category_hierarchy.is_file()
        ), f"Path to input categories is erroneous, check {self.path2category_hierarchy}. Current pwd: {os.getcwd()}"

    def _get_constraints_from_hierarchy_dict(self, hierarchy_dict: dict) -> List[str]:
        """
        Build all MLB triples (x, y, z) over hierarchy nodes that satisfy:
          Property 1:  all clusters containing x and z also contain y
          Property 2:  there exists a cluster containing x and y, but not z

        We model "d in c" as: d is a descendant (or equal) of cluster node c.
        For a node `n`, let Anc(n) be the set of ancestors of `n` (including `n` itself).
        Then:
          Property 1 <=> Anc(x) ∩ Anc(z) ⊆ Anc(y)
          Property 2 <=> (Anc(x) ∩ Anc(y)) \\ Anc(z) != ∅

        If `hierarchy_dict` is a forest (multiple roots), a synthetic root is added
        internally to connect top-level roots. The synthetic root is never emitted as
        a constraint item.
        """
        assert isinstance(hierarchy_dict, dict), "hierarchy_dict must be a dict."
        assert hierarchy_dict, "hierarchy_dict must not be empty."

        children_by_parent = {}
        parents_by_child = defaultdict(set)
        real_nodes = set()

        for parent, children in hierarchy_dict.items():
            assert isinstance(parent, (str, frozenset))
            assert isinstance(children, (list, set, tuple))
            assert all(isinstance(child, (str, frozenset)) for child in children)

            normalized_children = set(children)
            children_by_parent[parent] = normalized_children

            real_nodes.add(parent)
            real_nodes.update(normalized_children)

            for child in normalized_children:
                parents_by_child[child].add(parent)

        if len(real_nodes) < 3:
            return []

        for node in real_nodes:
            parents_by_child.setdefault(node, set())

        # Support forests by adding one synthetic super-root above top-level roots.
        synthetic_root = "__synthetic_root__"
        while synthetic_root in real_nodes:
            synthetic_root += "_"
        roots = [node for node in real_nodes if len(parents_by_child[node]) == 0]
        if roots:
            for root in roots:
                parents_by_child[root].add(synthetic_root)
            parents_by_child[synthetic_root] = set()

        ancestors_cache = {}
        visiting = set()

        def ancestors(node):
            if node in ancestors_cache:
                return ancestors_cache[node]
            if node in visiting:
                raise AssertionError(f"Cycle detected in hierarchy at node '{node}'.")
            visiting.add(node)
            result = {node}
            for parent in parents_by_child.get(node, set()):
                result |= ancestors(parent)
            visiting.remove(node)
            ancestors_cache[node] = result
            return result

        # Precompute ancestor closures for all emitted nodes and auxiliary nodes.
        all_nodes_for_ancestry = set(real_nodes)
        if roots:
            all_nodes_for_ancestry.add(synthetic_root)
        ancestors_by_node = {node: ancestors(node) for node in all_nodes_for_ancestry}

        # Enumerate unordered (x, y) pairs only once (MLB is symmetric in x/y).
        labels = {node: self._node_to_label(node) for node in real_nodes}
        sorted_nodes = sorted(real_nodes, key=lambda node: labels[node])
        constraints_set = set()

        for idx, x in enumerate(sorted_nodes):
            ancestors_x = ancestors_by_node[x]
            label_x = labels[x]
            for y in sorted_nodes[idx + 1 :]:
                ancestors_y = ancestors_by_node[y]
                label_y = labels[y]
                common_xy = ancestors_x & ancestors_y
                if not common_xy:
                    continue
                for z in sorted_nodes:
                    if z == x or z == y:
                        continue
                    ancestors_z = ancestors_by_node[z]

                    # Property 2: there exists c with x,y in c and z not in c.
                    if not (common_xy - ancestors_z):
                        continue
                    # Property 1: every c containing x,z also contains y.
                    # remove clusters containing y from clusters containing x and z -> if anything left: property
                    # violated
                    if (ancestors_x & ancestors_z) - ancestors_y:
                        continue
                    # Symmetric Property 1: (y,z) -> x
                    if (ancestors_y & ancestors_z) - ancestors_x:
                        continue
                    constraints_set.add(f"{label_x},{label_y},{labels[z]}")

        return sorted(constraints_set)

    def extract_all_mlb_constraints(self, out_path: Path):
        """
        Extract all MLB constraints from hierarchy dictionary.
        A triple (x, y, z) is emitted iff MLB Properties 1 and 2 hold.

        :param out_path: Path to save MLB constraints to (as txt file).

        Input:
          `out_path`: directory path. Example: Path("resources/banksearch/topic_model/").

        Output:
          Writes a txt file `mlb_banksearch.txt` containing lines like:
          Commercial Banks,Building Societies,Science
          Astronomy,Biology,Finance
        """
        assert isinstance(out_path, Path), "out_path must be a pathlib.Path."
        out_path.mkdir(parents=True, exist_ok=True)
        out_filename = out_path / f"mlb_{self.dataset_name}.txt"

        logger.info(f"Loading hierarchy from {self.path2category_hierarchy}")
        with open(self.path2category_hierarchy, "r") as f:
            hierarchy_dict = json.load(f)

        assert isinstance(hierarchy_dict, dict), "category_hierarchy.json must contain a JSON object."
        assert hierarchy_dict, "category_hierarchy.json must not be empty."
        constraints = self._get_constraints_from_hierarchy_dict(hierarchy_dict)
        assert constraints, "no constraints generated; check hierarchy structure."
        logger.info(f"Saving {len(constraints)} constraints to {out_filename}")
        with open(out_filename, "w") as f:
            for constraint in constraints:
                f.write(constraint + "\n")

        logger.info("Constraints successfully saved.")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    banksearch_extractor = BankSearchGroundTruthExtractor()
    out_path = Path(str(PROJECT_ROOT) + "/resources/banksearch/ground_truth/")
    out_path.mkdir(parents=True, exist_ok=True)
    banksearch_extractor.extract_all_mlb_constraints(out_path=out_path)
