import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_PATH)
from constraints.extractor import BankSearchTopicModelExtractor


class TestDomainExpertConstraints(unittest.TestCase):
    def test_get_constraints_from_domain_expert_minimal(self):
        # Minimal lattice:
        # top extent contains all docs, intent is empty.
        # Hierarchy:
        #       top
        #     /  |  \
        #    a   b   c
        concepts = [
            (["a", "b", "c"], []),
            (["a"], ["t1"]),
            (["b"], ["t2"]),
            (["c"], ["t3"]),
        ]
        extractor = BankSearchTopicModelExtractor(iceberg_concepts=concepts)
        constraints = extractor._get_constraints_from_domain_expert()

        # Parse constraints as unordered (a,b) pairs with c.
        parsed = set()
        for line in constraints:
            a, b, c = line.split(",")
            parsed.add((tuple(sorted([a, b])), c))

        expected = {
            (("a", "b"), "c"),
            (("a", "c"), "b"),
            (("b", "c"), "a"),
        }
        self.assertEqual(parsed, expected)

    def test_get_constraints_from_hierarchy_dict_uncle_rule(self):
        # Hierarchy:
        #       1
        #     /   \
        #    2     3
        #         / \
        #        4   5
        hierarchy = {
            "1": {"2", "3"},
            "3": {"4", "5"},
        }
        extractor = BankSearchTopicModelExtractor(iceberg_concepts=[])
        constraints = extractor._get_constraints_from_hierarchy_dict(hierarchy)

        # Only (4,5,2) should appear.
        self.assertEqual(constraints, ["4,5,2"])


if __name__ == "__main__":
    unittest.main()
