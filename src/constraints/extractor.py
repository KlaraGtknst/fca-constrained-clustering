from abc import ABC
from collections import defaultdict
import os
from pathlib import Path
from typing import List, Union
import json
from itertools import combinations
import logging

import edn_format

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

    def _get_mlb_constraints(self, parent: Union[str, frozenset], children: Union[List[Union[str, frozenset]], set[Union[str, frozenset]]]) -> List[str]:
        """
        Generate all pairwise combinations of children with parent (do not consider order of childern).

        Input:
          `parent`: label/ID for the parent concept, e.g. "Savings" or "C3".
          `children`: iterable of child labels/IDs, e.g. ["Checking", "Loans", "Investments"] or {"C4", "C5"}.

        Output:
          List of MLB constraint strings in "child1,child2,parent" format, e.g.
          ["Checking,Loans,Savings", "Checking,Investments,Savings", "Loans,Investments,Savings"]
          or with IDs:
          ["C4,C5,C3", "C4,C6,C3"].
        """
        assert isinstance(parent, (str, frozenset)), f"parent must be a string label, but got {type(parent)}."
        assert isinstance(children, (list, set)), f"children must be a list/set of string labels, but got {type(children)}."
        assert all(isinstance(c, (str, frozenset)) for c in children), f"each child label must be a string, but got {[type(c) for c in children]}."
        assert len(children) >= self.min_num_children, f"need at least {self.min_num_children} children to form pairwise constraints, got {len(children)}: {children}."

        # FIXME: How to work with one-element children? I suggest a: {b} -> (b,a,a) constraint which enforces b to be clustered with a
        if len(children) == 1:
            c1 = next(iter(children))
            logger.warning(f"Only one child {c1} for parent {parent}; generating single-child constraint.")
            return [f"{c1},{parent},{parent}"]

        # implicit concepts without explicit labels (e.g., conjunctions of multiple attributes) should be (already) replaced by unique IDs before this point
        assert "," not in parent, "parent label must not contain commas."
        assert all("," not in c for c in children), "child labels must not contain commas."
        return [f"{c1},{c2},{parent}" for c1, c2 in combinations(children, 2)]

    def _get_constraints_from_hierarchy_dict(self, hierarchy_dict: dict) -> List[str]:
        """
        Build MLB constraints from a parent -> children mapping.

        Input:
          `hierarchy_dict`: dict mapping parent labels/IDs to iterable child labels/IDs, e.g.
          {"Savings": ["Checking", "Loans"]} or {"C3": {"C4", "C5"}}.
          The parent is above the children in the concept hierarchy.

        Output:
          Flat list of "child1,child2,parent" constraint strings.
        """
        assert isinstance(hierarchy_dict, dict), "hierarchy_dict must be a dict."
        assert len(hierarchy_dict) > 0, "hierarchy_dict must not be empty."
        constraints = []
        for parent, children in hierarchy_dict.items():
            assert isinstance(parent, (str, frozenset)), f"hierarchy_dict keys must be strings or frozensets, but got {type(parent)}."
            assert isinstance(children, (list, set, tuple)), f"hierarchy_dict values must be iterables, but got {type(children)}."
            if len(children) < self.min_num_children:
                logger.warning(f"Parent {parent} has less than {self.min_num_children} children; skipping constraint generation; children: {children}.")
                continue
            constraints.extend(
                self._get_mlb_constraints(parent=parent, children=children)
            )

        return constraints


class BankSearchGroundTruthExtractor(BaseExtractor):

    def __init__(self):
        """
        Initialize the ground-truth extractor for the BankSearch dataset.

        Input:
          None.

        Output:
          Sets `self.path2category_hierarchy` to:
          "resources/banksearch/category_hierarchy.json".

        Expected input file structure:
          A JSON object mapping parent label -> list of child labels, e.g.
          {
            "Accounts": ["Checking", "Savings"],
            "Loans": ["Mortgage", "Personal"]
          }
        """
        super().__init__(dataset_name="banksearch")
        self.path2category_hierarchy = Path(
            "resources/banksearch/category_hierarchy.json"
        )
        assert (
            self.path2category_hierarchy.exists()
            and self.path2category_hierarchy.is_file()
        ), f"Path to input categories is erroneous, check {self.path2category_hierarchy}."

    def extract_all_mlb_constraints(self, out_path: Path):
        """
        Extract all MLB constraints from hierarchy dictionary.
        Only works for flat hierarchies so far.

        :param out_path: Path to save MLB constraints to (as txt file).

        Input:
          `out_path`: directory path. Example: Path("resources/banksearch/topic_model/").

        Output:
          Writes a txt file `mlb_banksearch.txt` containing lines like:
          Checking,Loans,Accounts
          Savings,Checking,Accounts
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


class BankSearchTopicModelExtractor(BaseExtractor):

    def __init__(self, iceberg_concepts: List):
        """
        Initialize the topic-model extractor.

        Input:
          `iceberg_concepts`: list of FCA concepts in the form (extent, intent).
          Example:
            [
              (["doc1", "doc2"], ["topicA", "topicB"]),
              (["doc2"], ["topicA"])
            ]
          where extent = list of document IDs and intent = list of topic labels.

        Output:
          Sets `self.iceberg_concepts`.
        """
        assert isinstance(iceberg_concepts, (list, edn_format.immutable_list.ImmutableList)), f"iceberg_concepts must be a list, but got {type(iceberg_concepts)}."
        super().__init__(dataset_name="banksearch")
        self.iceberg_concepts = iceberg_concepts

    def _iter_valid_concepts(self):
        """
        Iterate over iceberg concepts with validation and normalization.

        Input:
          `self.iceberg_concepts`: list-like of concepts in (extent, intent) form.

        Output:
          Yields `(extent_set, intent_set)` where both are Python sets.

        Skips:
          Concepts that are None, too short, of the wrong container type, or
          have non-iterable extent/intent members.
        """
        for concept in self.iceberg_concepts:
            # extent, intent
            if not concept or len(concept) < 2:
                continue
            if not isinstance(
                concept, (list, tuple, edn_format.immutable_list.ImmutableList)
            ):
                logger.warning(
                    "Skipping concept: expected tuple/list-like (extent, intent) but got %s.",
                    type(concept),
                )
                continue
            extent_raw, intent_raw = concept[0], concept[1]
            if not hasattr(extent_raw, "__iter__") or not hasattr(intent_raw, "__iter__"):
                logger.warning(
                    "Skipping concept: extent/intent must be iterable, got extent=%s intent=%s.",
                    type(extent_raw),
                    type(intent_raw),
                )
                continue
            extent_set = set(extent_raw)
            intent_set = set(intent_raw)
            yield extent_set, intent_set

    @staticmethod
    def _get_sorted_str_of_frozenset(fs: Union[frozenset, str]) -> str:
        """
        Canonical string key for a conjunction of labels.

        Input:
          `concept`: frozenset of labels, e.g. frozenset({"topicA", "topicB"}).

        Output:
          Comma-joined, sorted string, e.g. "topicA,topicB".
        """
        if isinstance(fs, str):
            return fs
        assert isinstance(fs, frozenset), f"Input must be a frozenset, but got {type(fs)}."
        return ",".join(sorted(fs))

    @staticmethod
    def _atomic_label(concept: frozenset) -> str:
        """
        Extract a label from an atomic concept.

        Input:
          `concept`: frozenset with 0 or 1 elements.
          Example: frozenset({"topicA"}) or frozenset().

        Output:
          "" for empty, or the single label as a string.
        """
        assert isinstance(concept, frozenset), f"concept must be a frozenset, but got {type(concept)}."
        if not concept:
            return ""
        if len(concept) == 1:
            return next(iter(concept))
        raise ValueError("Expected atomic concept with length 0 or 1.")

    def extract_all_mlb_constraints(self, out_path: Path):
        """
        Extract all MLB constraints from hierarchy in iceberg concepts.

        :param out_path: Path to save MLB constraints to (as txt file).

        Input:
          `out_path`: directory path to write files to.

        Output:
          Writes:
          `mlb_topic_model_banksearch.txt` with MLB constraints in ID form.
          Each label is an intent ID (C*), except the empty intent which is "top".
          `mlb_topic_model_banksearch_ids_map.json` mapping IDs to their intent conjunction strings.

          Example constraint line in txt file:
            C2,C5,C1
          Example JSON entry:
            "C5": "topicC,topicD"
            "top": ""
        """
        assert isinstance(out_path, Path), "out_path must be a pathlib.Path."
        assert self.iceberg_concepts, "iceberg_concepts is empty."
        out_path.mkdir(parents=True, exist_ok=True)
        out_filename = out_path / f"mlb_topic_model_{self.dataset_name}.txt"

        # hierarchy_dict: parent intent ID is key, set of child intent IDs is value
        hierarchy_dict = defaultdict(set)
        translate_intents = {}  # map frozenset(intent) -> ID (or "top" for empty)
        intent_id_counter = 1
        normalized_concepts = list(self._iter_valid_concepts())
        assert normalized_concepts, "no valid concepts after validation."

        def get_intent_id(intent_set):
            """
            Map an intent set to a stable label:
              - empty set -> "top"
              - singleton or conjunction -> unique C* ID

            Input:
              `intent_set`: set of topic labels, e.g. {"topicA"} or {"topicA","topicB"}.

            Output:
              String ID for the intent, e.g. "C3" (or "top" for empty).
            """
            nonlocal intent_id_counter
            if len(intent_set) == 0:
                key = frozenset()
                if key in translate_intents and translate_intents[key] != "top":
                    raise ValueError("Conflicting mapping for empty intent; expected 'top'.")
                translate_intents[key] = "top"
                return "top"
            key = self._get_sorted_str_of_frozenset(frozenset(intent_set))
            if key not in translate_intents:
                translate_intents[key] = f"C{intent_id_counter}"
                intent_id_counter += 1
            return translate_intents[key]

        for extent_set, intent_set in normalized_concepts:
            # documents are objects, topics are attributes in FCA terminology
            # i want MLB constraints on topics, so on intent
            # concept (A, B): A extent (objects), B  intent (attributes)
            # extent_set: all documents sharing the intent
            # intent_set: all topics shared by the extent
            intent_key = get_intent_id(intent_set)

            if intent_key in hierarchy_dict.keys():
                continue    # already processed
            for other_extent_set, other_intent_set in normalized_concepts:
                if extent_set == other_extent_set:
                    continue
                # check relation over extents subset-relation (documents)
                if other_extent_set.issubset(extent_set):
                    # Require intent_set ⊆ other_intent_set so children are true specializations (should always hold in FCA)
                    assert intent_set.issubset(other_intent_set) and intent_set != other_intent_set, "Invalid concept hierarchy: parent intent must be a proper subset of child intent."
                    other_intent_key = get_intent_id(other_intent_set)
                    hierarchy_dict[intent_key].add(other_intent_key)
                    

        def assert_acyclic(graph):
            visiting = set()
            visited = set()

            def dfs(node, stack):
                if node in visiting:
                    cycle = " -> ".join(stack + [node])
                    raise AssertionError(f"Cycle detected: {cycle}")
                if node in visited:
                    return
                visiting.add(node)
                for child in graph.get(node, set()):
                    if child in graph:  # only traverse nodes that are also parents
                        dfs(child, stack + [node])
                visiting.remove(node)
                visited.add(node)

            for node in graph:
                dfs(node, [])

        assert_acyclic(hierarchy_dict)

        # hierarchy_dict currently contains transitive edges, which leads to more constraints than necessary.
        # We remove transitive edges: if parent -> child and parent -> grandchild via child,
        # then drop parent -> grandchild.
        def all_descendants(node, graph, memo):
            if node in memo:
                return memo[node]
            desc = set()
            for ch in graph.get(node, set()):
                desc.add(ch)
                desc |= all_descendants(ch, graph, memo)
            memo[node] = desc
            return desc

        graph_snapshot = {k: set(v) for k, v in hierarchy_dict.items()}
        memo = {}
        for parent in list(hierarchy_dict.keys()):
            children = set(hierarchy_dict[parent])
            to_remove = set()
            for child in children:
                # Remove descendants of the child (not the child itself).
                to_remove |= all_descendants(child, graph_snapshot, memo) - {child}
            if to_remove:
                logger.info(
                    "Removing %d transitive children from parent '%s'.", len(to_remove), parent
                )
            hierarchy_dict[parent] = children - to_remove

        logger.info(
            f"Extracted hierarchy dict: {hierarchy_dict} of len {len(hierarchy_dict)} without any cycles."
        )
        assert hierarchy_dict, "hierarchy_dict is empty; no constraints can be formed."

        # Convert hierarchy dict into MLB constraints (child1, child2, parent).
        constraints = self._get_constraints_from_hierarchy_dict(hierarchy_dict)
        assert constraints, "no constraints generated; check iceberg_concepts content."
        logger.info(f"Got {len(constraints)} constraints: {constraints}")
        with open(out_filename, "w") as f:
            for constraint in constraints:
                f.write(constraint + "\n")
        logger.info(f"Saved {len(constraints)} constraints to {out_filename}")

    
        out_filename_ids_map = (
            out_path / f"mlb_topic_model_{self.dataset_name}_ids_map.json"
        )
        
        with open(out_filename_ids_map, "w") as f:
            json.dump(
                {v: self._get_sorted_str_of_frozenset(k) for k, v in translate_intents.items()},
                f,
                indent=2,
                sort_keys=True,
            )
        logger.info(
            f"Saved {len(translate_intents)} intent identifier mappings to {out_filename_ids_map}"
        )

if __name__ == "__main__":
    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    banksearch_extractor = BankSearchGroundTruthExtractor()
    out_path = Path("resources/banksearch/topic_model/")
    out_path.mkdir(parents=True, exist_ok=True)
    banksearch_extractor.extract_all_mlb_constraints(out_path=out_path)
