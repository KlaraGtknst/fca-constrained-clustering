import os
import sys
import unittest
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_PATH)

from constraints.extractor import BankSearchGroundTruthExtractor


class TestGroundTruthConstraints(unittest.TestCase):
    def setUp(self):
        self.extractor = BankSearchGroundTruthExtractor()

    def test_get_mlb_constraints_sibling_uncle_shape(self):
        constraints = self.extractor._get_mlb_constraints(
            third_node="uncle",
            siblings={"child_b", "child_a"},
        )
        self.assertEqual(constraints, ["child_a,child_b,uncle"])

    def test_get_mlb_constraints_skips_single_sibling(self):
        constraints = self.extractor._get_mlb_constraints(
            third_node="uncle",
            siblings={"child_a"},
        )
        self.assertEqual(constraints, [])

    @staticmethod
    def _ancestors_by_node(hierarchy: dict[str, set[str] | list[str] | tuple[str, ...]]) -> dict[str, set[str]]:
        children_by_parent = {parent: set(children) for parent, children in hierarchy.items()}
        real_nodes = set(children_by_parent.keys())
        for children in children_by_parent.values():
            real_nodes.update(children)

        parents_by_child = defaultdict(set)
        for parent, children in children_by_parent.items():
            for child in children:
                parents_by_child[child].add(parent)
        for node in real_nodes:
            parents_by_child.setdefault(node, set())

        synthetic_root = "__synthetic_root_for_test__"
        while synthetic_root in real_nodes:
            synthetic_root += "_"
        roots = [node for node in real_nodes if len(parents_by_child[node]) == 0]
        for root in roots:
            parents_by_child[root].add(synthetic_root)
        parents_by_child[synthetic_root] = set()

        cache = {}
        visiting = set()

        def ancestors(node: str) -> set[str]:
            if node in cache:
                return cache[node]
            if node in visiting:
                raise AssertionError(f"Cycle detected in hierarchy at node '{node}'.")
            visiting.add(node)
            result = {node}
            for parent in parents_by_child[node]:
                result |= ancestors(parent)
            visiting.remove(node)
            cache[node] = result
            return result

        return {node: ancestors(node) for node in real_nodes}

    @staticmethod
    def _satisfies_mlb_properties(ancestors: dict[str, set[str]], x: str, y: str, z: str) -> bool:
        common_xy = ancestors[x] & ancestors[y]

        # Property 2: exists c with x,y in c and z not in c.
        property_2 = bool(common_xy - ancestors[z])
        if not property_2:
            return False

        # Property 1: all c containing x,z also contain y.
        if (ancestors[x] & ancestors[z]) - ancestors[y]:
            return False
        # Symmetric Property 1: all c containing y,z also contain x.
        if (ancestors[y] & ancestors[z]) - ancestors[x]:
            return False

        return True

    def test_ground_truth_hierarchy_dict_matches_property_logic_exactly(self):
        hierarchy = {
            "root": {"a", "b", "c"},
            "a": {"a1", "a2"},
            "b": {"b1", "b2"},
        }

        constraints = self.extractor._get_constraints_from_hierarchy_dict(hierarchy)
        produced = {tuple(line.split(",")) for line in constraints}

        ancestors = self._ancestors_by_node(hierarchy)
        nodes = sorted(ancestors.keys())
        expected = set()
        for i, x in enumerate(nodes):
            for y in nodes[i + 1 :]:
                for z in nodes:
                    if z == x or z == y:
                        continue
                    if self._satisfies_mlb_properties(ancestors, x, y, z):
                        expected.add((x, y, z))

        self.assertEqual(produced, expected)


if __name__ == "__main__":
    unittest.main()
