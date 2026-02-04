import unittest

from constraints.extractor import BankSearchTopicModelExtractor


class TestDomainExpertConstraints(unittest.TestCase):
    def test_get_constraints_from_domain_expert_minimal(self):
        # Minimal lattice:
        # top extent contains all docs, intent is empty.
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


if __name__ == "__main__":
    unittest.main()
