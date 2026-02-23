import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_PATH)

from attribute_exploration.expand_mlb_cxt_equivalence import (  # noqa: E402
    _expand_rows,
    _read_burmeister,
    _write_burmeister,
)


class TestExpandMlbCxtEquivalence(unittest.TestCase):
    def test_expand_rows_copies_representative_row(self):
        objects = ["A0001", "B0001", "Finance"]
        rows_by_object = {
            "A0001": "X.",
            "B0001": ".X",
            "Finance": "..",
        }
        classes = {
            "A0001": ["A0001", "A0002"],
            "B0001": ["B0001", "B0002"],
        }

        expanded_objects, expanded_rows = _expand_rows(objects, rows_by_object, classes)

        expected = {
            "A0001": "X.",
            "A0002": "X.",
            "B0001": ".X",
            "B0002": ".X",
            "Finance": "..",
        }
        self.assertEqual(expanded_objects, list(expected.keys()))
        self.assertEqual(expanded_rows, list(expected.values()))

    def test_burmeister_roundtrip_with_expanded_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "nested" / "toy_expanded.cxt"
            objects = ["A0001", "A0002", "Finance"]
            attributes = ["M0", "M1"]
            rows = ["X.", "X.", ".."]

            _write_burmeister(out_path, objects, attributes, rows)
            loaded_objects, loaded_attributes, loaded_rows_by_object = _read_burmeister(out_path)

            self.assertEqual(loaded_objects, objects)
            self.assertEqual(loaded_attributes, attributes)
            self.assertEqual(loaded_rows_by_object, {o: r for o, r in zip(objects, rows)})


if __name__ == "__main__":
    unittest.main()
