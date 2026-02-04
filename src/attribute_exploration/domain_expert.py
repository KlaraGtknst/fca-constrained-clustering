import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from edn_format import loads

logger = logging.getLogger(__name__)


def read_edn_concepts(edn_path: str) -> List[Tuple[Set[str], Set[str]]]:
    """
    Read an EDN iceberg concept lattice and return normalized concepts.

    Input:
      `edn_path` (str): path to an EDN file that encodes concepts as
      `[(extent, intent), ...]`, where extent is a list of document IDs
      and intent is a list of attribute/topic labels.

    Output:
      List of `(extent_set, intent_set)` tuples where both sets contain strings.

    Example input (EDN-like):
      [
        (["doc1", "doc2"], ["topicA"]),
        (["doc2"], ["topicA", "topicB"])
      ]

    Example output:
      [
        ({"doc1", "doc2"}, {"topicA"}),
        ({"doc2"}, {"topicA", "topicB"})
      ]
    """
    with open(edn_path, "r", encoding="utf-8") as f:
        edn_data = loads(f.read())
    concepts: List[Tuple[Set[str], Set[str]]] = []
    for concept in edn_data:
        if not concept or len(concept) < 2:
            continue
        extent_raw, intent_raw = concept[0], concept[1]
        extent_set = set(map(str, extent_raw))
        intent_set = set(map(str, intent_raw))
        concepts.append((extent_set, intent_set))
    return concepts


def _sorted_key(items: Iterable[str]) -> Tuple[int, List[str]]:
    """Stable key for choosing a smallest set deterministically."""
    sorted_items = sorted(items)
    return (len(sorted_items), sorted_items)


@dataclass(frozen=True)
class ImplicationResult:
    """
    Result for a domain-expert implication query.

    Attributes:
      `is_true`: True if the implication holds, False otherwise.
      `counterexample`: None if true, else a pair of extents `[e_i, e_j]`
      that witness the violation.
    """

    is_true: bool
    counterexample: Optional[List[List[str]]]


class DomainExpert:
    """
    Domain expert backed by an iceberg concept lattice (EDN).

    This class answers implication queries of the form (d_x, d_y, d_z):
      - Let e_x be the lowest concept extent containing d_x.
      - Let e_y be the lowest concept extent containing d_y.
      - Let meet(e_x, e_y) be the lowest concept extent that contains all
        documents in e_x ∪ e_y.
      - The implication is True iff d_z ∈ meet(e_x, e_y).

    If False, returns a counterexample `[e_i, e_j]` where:
      - e_i is the lowest extent containing {d_x, d_z}
      - e_j is the lowest extent containing {d_y}
    (as suggested: e_i contains d_x and d_z; e_j contains d_y)
    """

    def __init__(self, concepts: List[Tuple[Set[str], Set[str]]]):
        """
        Initialize the domain expert from an EDN lattice file.

        Input:
          `concepts` (List[Tuple[Set[str], Set[str]]]): list of iceberg lattice concepts.

        Output:
          Loads and stores the lattice concepts as sets of strings.
        """
        self.concepts = concepts
        assert self.concepts, "No concepts loaded."
        self.extents = [extent for extent, _ in self.concepts]

    def _lowest_extent_containing(self, docs: Set[str]) -> Optional[Set[str]]:
        """
        Return the smallest (lowest) extent that contains all docs.

        Input:
          `docs` (set[str]): document IDs to be contained.

        Output:
          The minimal extent set that contains `docs`, or None if no extent matches.
        """
        candidates = [e for e in self.extents if docs.issubset(e)]
        if not candidates:
            return None
        # Choose the smallest extent; break ties deterministically.
        return min(candidates, key=_sorted_key)

    def implies(self, d_x: str, d_y: str, d_z: str) -> ImplicationResult:
        """
        Check whether the implication (d_x, d_y, d_z) holds.

        Input:
          `d_x`, `d_y`, `d_z` (str): document IDs.

        Output:
          ImplicationResult:
            - is_true=True, counterexample=None if d_z is in the meet.
            - is_true=False, counterexample=[e_i, e_j] if not.

        Example:
          If e_x={"d1","d2","d4"}, e_y={"d2","d3"}, meet={"d1","d2","d3","d4"}:
            implies("d1","d3","d2") -> True
            implies("d2","d3","d4") -> False, counterexample=[["d2","d3"],["d4"]]
        """
        assert isinstance(d_x, str) and isinstance(d_y, str) and isinstance(d_z, str), (
            "d_x, d_y, d_z must be strings."
        )
        e_x = self._lowest_extent_containing({d_x})
        e_y = self._lowest_extent_containing({d_y})
        if e_x is None or e_y is None:
            return ImplicationResult(False, None)

        meet = self._lowest_extent_containing(set(e_x) | set(e_y))
        if meet is None:
            return ImplicationResult(False, None)

        if d_z in meet:
            return ImplicationResult(True, None)

        # Counterexample per instruction:
        e_j = self._lowest_extent_containing(meet | {d_x, d_y, d_z})
        if meet is None or e_j is None:
            return ImplicationResult(False, None)
        return ImplicationResult(False, [sorted(meet), sorted(e_j)])
