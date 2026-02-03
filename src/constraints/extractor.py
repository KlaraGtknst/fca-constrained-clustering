from abc import ABC
import os
from pathlib import Path
from typing import List
import json
from itertools import combinations
import logging

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def extract_all_mlb_constraints(self, out_path: Path):
        pass

    def _get_mlb_constraints(self, parent: str, children: List[str]) -> List[str]:
        """
        Generate all pairwise combinations of children with parent included (do not consider order of childern).
        """
        return [f"{c1},{c2},{parent}" for c1, c2 in combinations(children, 2)]

    def _get_constraints_from_hierarchy_dict(self, hierarchy_dict: dict) -> List[str]:
        constraints = []
        for parent, children in hierarchy_dict.items():
            constraints.extend(
                self._get_mlb_constraints(parent=parent, children=children)
            )

        return constraints


class BankSearchGroundTruthExtractor(BaseExtractor):

    def __init__(self):
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
        """
        out_path.mkdir(parents=True, exist_ok=True)
        out_filename = out_path / f"mlb_{self.dataset_name}.txt"

        logger.info(f"Loading hierarchy from {self.path2category_hierarchy}")
        with open(self.path2category_hierarchy, "r") as f:
            hierarchy_dict = json.load(f)

        constraints = self._get_constraints_from_hierarchy_dict(hierarchy_dict)
        logger.info(f"Saving {len(constraints)} constraints to {out_filename}")
        with open(out_filename, "w") as f:
            for constraint in constraints:
                f.write(constraint + "\n")

        logger.info("Constraints successfully saved.")


class BankSearchTopicModelExtractor(BaseExtractor):

    def __init__(self, iceberg_concepts: List):
        super().__init__(dataset_name="banksearch")
        self.iceberg_concepts = iceberg_concepts

    @staticmethod
    def _conjunction_key(concept: frozenset) -> str:
        return ",".join(sorted(concept))

    @staticmethod
    def _atomic_label(concept: frozenset) -> str:
        if not concept:
            return ""
        if len(concept) == 1:
            return next(iter(concept))
        raise ValueError("Expected atomic concept with length 0 or 1.")

    def extract_all_mlb_constraints(self, out_path: Path):
        """
        Extract all MLB constraints from hierarchy in iceberg concepts.

        :param out_path: Path to save MLB constraints to (as txt file).
        """
        out_path.mkdir(parents=True, exist_ok=True)
        out_filename = out_path / f"mlb_topic_model_{self.dataset_name}.txt"

        hierarchy_dict = {}
        for concept in self.iceberg_concepts:
            # logger.info("Processing concept:", concept)
            # extent, intent
            if not concept or len(concept) < 2:
                continue
            # documents are objects, topics are attributes in FCA terminology
            # i want MLB constraints on topics, so on intent
            # concept (A, B): A extent (objects), B  intent (attributes)
            intent_raw = concept[1]  # all attributes shared by the extent A
            extent_raw = concept[0]  # all objects having the attributes in intent B
            intent_set = set(intent_raw)
            extent_set = set(extent_raw)
            intent_key = frozenset(intent_set)
            if intent_key in hierarchy_dict.keys():
                continue
            hierarchy_dict[intent_key] = set()
            for other_concept in self.iceberg_concepts:
                if not other_concept or len(other_concept) < 2:
                    continue
                other_extent_raw = other_concept[0]
                other_extent_set = set(other_extent_raw)
                if extent_set == other_extent_set:
                    continue
                if other_extent_set.issubset(extent_set):
                    # TODO: how to handle not-explicitly represented concepts? Those that are only unions of more specific concepts
                    hierarchy_dict[intent_key].add(frozenset(other_concept[1]))

        logger.info(
            f"Extracted hierarchy dict: {hierarchy_dict} of len {len(hierarchy_dict)}"
        )

        constraints = self._get_constraints_from_hierarchy_dict(hierarchy_dict)
        logger.info(f"Saving {len(constraints)} constraints to {out_filename}")
        with open(out_filename, "w") as f:
            for constraint in constraints:
                # Convert frozensets (e.g., frozenset({'7'}),frozenset({'2'}),frozenset()) back to normal sets or lists (e.g., "7","2", "")
                constraint_str = constraint.replace("frozenset({", "").replace("})", "")
                constraint_str = constraint_str.replace("frozenset(", '"').replace(
                    ")", '"'
                )
                constraint_str = constraint_str.replace("'", "")
                f.write(constraint_str + "\n")

        # FIXME: "" at third position gets lost in the below writing process, need to recover it
        # Build constraints with explicit identifiers for unnamed conjunctions.
        # Reason: When a concept is only implicitly represented (as a conjunction of
        # multiple attributes), the stringified constraint becomes long (e.g., "x,y").
        # We generate stable, unused identifiers (CJ*) to keep constraints compact and
        # save a JSON map so the original conjunctions can be recovered.
        constraints_raw = []
        for parent, children in hierarchy_dict.items():
            for c1, c2 in combinations(children, 2):
                constraints_raw.append((c1, c2, parent))

        used_labels = set()
        for concept in hierarchy_dict.keys():
            used_labels.update(concept)
        for children in hierarchy_dict.values():
            for concept in children:
                used_labels.update(concept)

        conjunctions = set()
        for c1, c2, parent in constraints_raw:
            if len(c1) > 1:
                conjunctions.add(c1)
            if len(c2) > 1:
                conjunctions.add(c2)
            if len(parent) > 1:
                conjunctions.add(parent)

        conjunction_id_map = {}
        counter = 1
        for conj in sorted(conjunctions, key=self._conjunction_key):
            conj_key = self._conjunction_key(conj)
            conj_id = f"CJ{counter}"
            while conj_id in used_labels:
                counter += 1
                conj_id = f"CJ{counter}"
            conjunction_id_map[conj] = conj_id
            used_labels.add(conj_id)
            counter += 1

        out_filename_ids = out_path / f"mlb_topic_model_{self.dataset_name}_ids.txt"
        logger.info(
            f"Saving {len(constraints_raw)} constraints with identifiers to {out_filename_ids}"
        )
        with open(out_filename_ids, "w") as f:
            for c1, c2, parent in constraints_raw:
                if len(c1) > 1:
                    c1_label = conjunction_id_map[c1]
                else:
                    c1_label = self._atomic_label(c1)

                if len(c2) > 1:
                    c2_label = conjunction_id_map[c2]
                else:
                    c2_label = self._atomic_label(c2)

                if len(parent) > 1:
                    parent_label = conjunction_id_map[parent]
                else:
                    parent_label = self._atomic_label(parent)

                f.write(f"{c1_label},{c2_label},{parent_label}\n")

        out_filename_ids_map = (
            out_path / f"mlb_topic_model_{self.dataset_name}_ids_map.json"
        )
        logger.info(
            f"Saving {len(conjunction_id_map)} conjunction identifier mappings to {out_filename_ids_map}"
        )
        with open(out_filename_ids_map, "w") as f:
            json.dump(
                {v: self._conjunction_key(k) for k, v in conjunction_id_map.items()},
                f,
                indent=2,
                sort_keys=True,
            )

        logger.info(
            f"Constraints successfully saved to {out_filename} and {out_filename_ids}."
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
