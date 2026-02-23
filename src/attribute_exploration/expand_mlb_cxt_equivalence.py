import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple


LOGGER = logging.getLogger(__name__)


def _resolve_path(path_str: str, project_root: Path) -> Path:
    path = Path(path_str)
    if path.is_file():
        return path
    candidate = project_root / path
    return candidate if candidate.is_file() else path


def _resolve_output_path(path_str: str, project_root: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else project_root / path


def _read_burmeister(path: Path) -> Tuple[List[str], List[str], Dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "B":
        raise ValueError(f"{path} does not look like a Burmeister .cxt file (missing 'B' header).")

    idx = 1
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines):
        raise ValueError(f"{path} is malformed: missing object count.")
    object_count = int(lines[idx].strip())

    idx += 1
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines):
        raise ValueError(f"{path} is malformed: missing attribute count.")
    attribute_count = int(lines[idx].strip())

    idx += 1
    while idx < len(lines) and not lines[idx].strip():
        idx += 1

    objects = lines[idx : idx + object_count]
    if len(objects) != object_count:
        raise ValueError(f"{path} is malformed: expected {object_count} objects, got {len(objects)}.")
    idx += object_count

    attributes = lines[idx : idx + attribute_count]
    if len(attributes) != attribute_count:
        raise ValueError(f"{path} is malformed: expected {attribute_count} attributes, got {len(attributes)}.")
    idx += attribute_count

    matrix_rows = lines[idx : idx + object_count]
    if len(matrix_rows) != object_count:
        raise ValueError(f"{path} is malformed: expected {object_count} incidence rows, got {len(matrix_rows)}.")

    for row_idx, row in enumerate(matrix_rows, start=1):
        if len(row) != attribute_count:
            raise ValueError(
                f"{path} is malformed: incidence row {row_idx} has length {len(row)}, expected {attribute_count}."
            )
        invalid = set(row) - {"X", "."}
        if invalid:
            raise ValueError(f"{path} is malformed: invalid symbols {sorted(invalid)} in incidence matrix.")

    return objects, attributes, {obj: row for obj, row in zip(objects, matrix_rows)}


def _write_burmeister(path: Path, objects: List[str], attributes: List[str], rows: List[str]) -> None:
    if len(objects) != len(rows):
        raise ValueError("Cannot write .cxt: object and row counts differ.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("B\n\n")
        handle.write(f"{len(objects)}\n")
        handle.write(f"{len(attributes)}\n\n")
        for obj in objects:
            handle.write(f"{obj}\n")
        for attribute in attributes:
            handle.write(f"{attribute}\n")
        for row in rows:
            handle.write(f"{row}\n")


def _load_equivalence_classes(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object mapping representative -> members.")

    parsed: Dict[str, List[str]] = {}
    for representative, members in data.items():
        if not isinstance(representative, str):
            LOGGER.warning("Skipping non-string representative key: %r", representative)
            continue
        if not isinstance(members, list) or not all(isinstance(member, str) for member in members):
            LOGGER.warning("Skipping representative '%s': members must be a list of strings.", representative)
            continue
        parsed[representative] = members
    return parsed


def _expand_rows(
    objects: List[str], rows_by_object: Dict[str, str], classes: Dict[str, List[str]]
) -> Tuple[List[str], List[str]]:
    expanded_objects: List[str] = []
    expanded_rows: List[str] = []
    assigned_member_to_rep: Dict[str, str] = {}

    for representative, members in classes.items():
        row = rows_by_object.get(representative)
        if row is None:
            LOGGER.warning("Representative '%s' not found in CXT object list; skipping class.", representative)
            continue
        if not members:
            LOGGER.warning("Representative '%s' has an empty equivalence class.", representative)
            continue

        for member in members:
            existing_rep = assigned_member_to_rep.get(member)
            if existing_rep is not None:
                if existing_rep != representative:
                    LOGGER.warning(
                        "Member '%s' appears in multiple classes ('%s' and '%s'); keeping first.",
                        member,
                        existing_rep,
                        representative,
                    )
                else:
                    LOGGER.warning(
                        "Duplicate member '%s' in class '%s'; keeping first occurrence.", member, representative
                    )
                continue

            assigned_member_to_rep[member] = representative
            expanded_objects.append(member)
            expanded_rows.append(row)

        if representative not in assigned_member_to_rep:
            LOGGER.warning(
                "Representative '%s' is missing from its own member list; appending it explicitly.",
                representative,
            )
            assigned_member_to_rep[representative] = representative
            expanded_objects.append(representative)
            expanded_rows.append(row)

    for obj in objects:
        if obj in assigned_member_to_rep:
            continue
        LOGGER.warning("Object '%s' has no equivalence class key; keeping original row.", obj)
        expanded_objects.append(obj)
        expanded_rows.append(rows_by_object[obj])

    return expanded_objects, expanded_rows


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description=(
            "Expand a Burmeister .cxt context so each member of an equivalence class "
            "gets the representative's incidence row."
        )
    )
    parser.add_argument(
        "--cxt",
        default="resources/banksearch/ground_truth/mlb.cxt",
        help="Path to input Burmeister .cxt file (default: resources/banksearch/ground_truth/mlb.cxt).",
    )
    parser.add_argument(
        "--equiv-json",
        default="resources/banksearch/ground_truth/mlb_banksearch_equivalence_classes.json",
        help="Path to JSON mapping representative -> class members.",
    )
    parser.add_argument(
        "--output",
        default="resources/banksearch/ground_truth/mlb_expanded.cxt",
        help="Path to output .cxt file. If omitted, input --cxt is overwritten in-place.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cxt_path = _resolve_path(args.cxt, project_root)
    equiv_path = _resolve_path(args.equiv_json, project_root)
    output_path = _resolve_output_path(args.output, project_root) if args.output else cxt_path

    if not cxt_path.is_file():
        raise FileNotFoundError(f"CXT file not found: {cxt_path}")
    if not equiv_path.is_file():
        raise FileNotFoundError(f"Equivalence class JSON file not found: {equiv_path}")

    LOGGER.info("Reading CXT from %s", cxt_path)
    objects, attributes, rows_by_object = _read_burmeister(cxt_path)
    LOGGER.info("Loaded %d objects and %d attributes.", len(objects), len(attributes))

    LOGGER.info("Reading equivalence classes from %s", equiv_path)
    classes = _load_equivalence_classes(equiv_path)
    LOGGER.info("Loaded %d equivalence class representatives.", len(classes))

    expanded_objects, expanded_rows = _expand_rows(objects, rows_by_object, classes)
    LOGGER.info(
        "Expanded context to %d objects (from %d) while keeping %d attributes.",
        len(expanded_objects),
        len(objects),
        len(attributes),
    )

    _write_burmeister(output_path, expanded_objects, attributes, expanded_rows)
    LOGGER.info("Wrote updated CXT to %s", output_path)


if __name__ == "__main__":
    main()
