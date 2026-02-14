import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set, Tuple

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
      - The implication is True iff
        - Property 1: for all concept extents e_xz with d_x, d_z in e_xz -> d_y in e_xz
        - and Property 2: not d_z ∈ meet(e_x, e_y)

    If False, returns a counterexample `[e_i, e_j]` where:
      - e_i is the lowest extent containing {d_x, d_z} and not d_y
      - e_j is the lowest extent containing {d_y}
    """

    def __init__(self, concepts: List[Tuple[Set[str], Set[str]]]):
        """
        Initialize the domain expert from an EDN lattice file.

        Input:
          `concepts` (List[Tuple[Set[str], Set[str]]]): list of iceberg lattice concepts.

        Output:
          Loads and stores the lattice concepts as sets of strings.
        """
        # Normalize to string sets so extent operations are always well-defined.
        normalized_concepts: List[Tuple[Set[str], Set[str]]] = []
        for concept in concepts:
            if not concept or len(concept) < 2:
                continue
            extent, intent = concept[0], concept[1]
            normalized_concepts.append(
                (set(map(str, extent)), set(map(str, intent)))
            )

        self.concepts = normalized_concepts
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
        return set(min(candidates, key=_sorted_key))

    def _lowest_extent_containing_without(
        self, required_docs: Set[str], forbidden_doc: str
    ) -> Optional[Set[str]]:
        """
        Return the smallest extent containing all required docs but excluding one doc.
        """
        candidates = [
            e
            for e in self.extents
            if required_docs.issubset(e) and forbidden_doc not in e
        ]
        if not candidates:
            return None
        return set(min(candidates, key=_sorted_key))

    def implies(self, d_x: str, d_y: str, d_z: str) -> ImplicationResult:
        """
        Check whether the implication (d_x, d_y, d_z) holds.

        Input:
          `d_x`, `d_y`, `d_z` (str): document IDs.

        Output:
          ImplicationResult:
            - is_true=True, counterexample=None iff:
              (1) every extent containing {d_x, d_z} also contains d_y, and
              (2) d_z is not in meet(e_x, e_y), where e_x/e_y are the lowest
                  extents containing d_x/d_y.
            - is_true=False otherwise. If available, counterexample=[e_i, e_j]
              with e_i the lowest extent containing {d_x, d_z} and excluding d_y,
              and e_j the lowest extent containing {d_y}.
        """
        assert isinstance(d_x, str) and isinstance(d_y, str) and isinstance(d_z, str), (
            "d_x, d_y, d_z must be strings."
        )
        # Property 1:
        #   for all extents e_xz with d_x, d_z in e_xz -> d_y in e_xz
        violating_extents = [
            e for e in self.extents if {d_x, d_z}.issubset(e) and d_y not in e
        ]
        property_1 = len(violating_extents) == 0

        # Property 2:
        #   d_z not in meet(e_x, e_y), where meet is the lowest extent
        #   containing e_x ∪ e_y.
        e_x = self._lowest_extent_containing({d_x})
        e_y = self._lowest_extent_containing({d_y})
        meet = (
            self._lowest_extent_containing(set(e_x) | set(e_y))
            if e_x is not None and e_y is not None
            else None
        )
        property_2 = meet is not None and d_z not in meet

        if property_1 and property_2:
            return ImplicationResult(True, None)

        e_i = self._lowest_extent_containing_without({d_x, d_z}, d_y)
        if not e_i:
            e_i = self._lowest_extent_containing_without({d_y, d_z}, d_x)
        e_j = self._lowest_extent_containing({d_y})
        if e_i is None or e_j is None:
            return ImplicationResult(False, None)
        return ImplicationResult(False, [sorted(e_i), sorted(e_j)])
