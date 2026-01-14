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

class TreeBankExtractor(BaseExtractor):

    def __init__(self):
        super().__init__(dataset_name="treebank")
        self.path2category_hierarchy = Path("resources/banksearch/category_hierarchy.json")
        assert self.path2category_hierarchy.exists() and self.path2category_hierarchy.is_file(), \
            f"Path to input categories is erroneous, check {self.path2category_hierarchy}."

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

        constraints = []
        for parent, children in hierarchy_dict.items():
            constraints.extend(self._get_mlb_constraints(parent=parent, children=children))

        logger.info(f"Saving {len(constraints)} constraints to {out_filename}")
        with open(out_filename, "w") as f:
            for constraint in constraints:
                f.write(constraint + "\n")

        logger.info("Constraints successfully saved.")

if __name__ == "__main__":
    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    treebank_extractor = TreeBankExtractor()
    out_path = Path("resources/banksearch")
    out_path.mkdir(parents=True, exist_ok=True)
    treebank_extractor.extract_all_mlb_constraints(out_path=out_path)