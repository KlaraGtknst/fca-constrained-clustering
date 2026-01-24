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

    def __init__(self, iceberg_concepts:List):
        super().__init__(dataset_name="banksearch")
        self.iceberg_concepts = iceberg_concepts

    def extract_all_mlb_constraints(self, out_path: Path):
        """
        Extract all MLB constraints from hierarchy in iceberg concepts.

        :param out_path: Path to save MLB constraints to (as txt file).
        """
        out_path.mkdir(parents=True, exist_ok=True)
        out_filename = out_path / f"mlb_topic_model_{self.dataset_name}.txt"

        constraints = []
        hierarchy_dict = {}
        for concept in self.iceberg_concepts:
            print("Processing concept:", concept)
            # extent, intent
            if not concept or len(concept) < 2:
                continue
            intent_raw = concept[1]
            extent_raw = concept[0]
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
                    hierarchy_dict[intent_key].add(frozenset(other_concept[1]))

        print("Extracted hierarchy dict:", hierarchy_dict)

        constraints = self._get_constraints_from_hierarchy_dict(hierarchy_dict)
        logger.info(f"Saving {len(constraints)} constraints to {out_filename}")
        with open(out_filename, "w") as f:
            for constraint in constraints:
                # Convert frozensets (e.g., frozenset({'7'}),frozenset({'2'}),frozenset()) back to normal sets or lists (e.g., "7","2", "")
                constraint_str = constraint.replace("frozenset({", "").replace("})", "")
                constraint_str = constraint_str.replace("frozenset(", '"').replace(")", '"')
                constraint_str = constraint_str.replace("'", '"')
                f.write(constraint_str + "\n")

        logger.info("Constraints successfully saved.")

if __name__ == "__main__":
    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    banksearch_extractor = BankSearchGroundTruthExtractor()
    out_path = Path("resources/banksearch")
    out_path.mkdir(parents=True, exist_ok=True)
    banksearch_extractor.extract_all_mlb_constraints(out_path=out_path)
