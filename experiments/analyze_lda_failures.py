"""Save BankSearch documents that are missing from the LDA FCA context.

The LDA pipeline writes processed document identifiers into the ``index`` field
of ``resources/banksearch/topic_model/fca_topic_model_context.json``. This
script compares those identifiers with the files available in
``resources/Dataset`` and writes one CSV file with the dataset documents that
have no matching context index entry.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "resources" / "Dataset"
DEFAULT_CONTEXT_PATH = (
    REPO_ROOT
    / "resources"
    / "banksearch"
    / "topic_model"
    / "fca_topic_model_context.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "LDA_fail_analysis"


def normalise_document_id(value: str) -> str:
    """Return the comparable document identifier for an index or file name."""
    return Path(value).stem


def load_context_document_ids(context_path: Path) -> set[str]:
    """Load document identifiers from the FCA topic-model context ``index``."""
    with context_path.open(encoding="utf-8") as context_file:
        context = json.load(context_file)

    index = context.get("index")
    if not isinstance(index, list):
        raise ValueError(f"Expected {context_path} to contain a list-valued 'index' entry.")

    # The context currently stores bare IDs, but normalising keeps the comparison
    # correct if a future export stores names such as "A0001.txt".
    return {normalise_document_id(str(entry)) for entry in index}


def read_dataset_value(document_path: Path) -> str:
    """Read the document's ``DATASET=...`` metadata value."""
    with document_path.open(encoding="utf-8", errors="replace") as document_file:
        for line in document_file:
            if line.startswith("DATASET="):
                return line.removeprefix("DATASET=").strip()
            if line.startswith("HTML="):
                break
    return ""


def iter_dataset_documents(dataset_dir: Path) -> Iterable[dict[str, str]]:
    """Yield the text documents available in the dataset directory."""
    for document_path in sorted(dataset_dir.glob("*.txt")):
        yield {
            "doc_id": normalise_document_id(document_path.name),
            "file_name": document_path.name,
            "dataset": read_dataset_value(document_path),
        }


def write_missing_documents_csv(output_path: Path, rows: list[dict[str, str]]) -> None:
    """Write the missing document rows with IDs, file names, and datasets."""
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["doc_id", "file_name", "dataset"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for rerunning the analysis with custom paths."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare resources/Dataset/*.txt against the topic-model FCA context "
            "index and save documents that are absent from the LDA output."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Directory containing dataset .txt files. Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--context-path",
        type=Path,
        default=DEFAULT_CONTEXT_PATH,
        help=f"FCA topic-model context JSON. Default: {DEFAULT_CONTEXT_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for analysis outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def main() -> None:
    """Run the missing-document analysis and write the requested CSV output."""
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    context_path = args.context_path.resolve()
    output_dir = args.output_dir.resolve()

    print(f"Reading dataset files from {dataset_dir}")
    dataset_documents = list(iter_dataset_documents(dataset_dir))

    print(f"Reading processed document IDs from {context_path}")
    context_document_ids = load_context_document_ids(context_path)

    # A document is considered unprocessed when its file stem is not present in
    # the topic-model context index.
    missing_rows = [
        document
        for document in dataset_documents
        if document["doc_id"] not in context_document_ids
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    missing_csv_path = output_dir / "missing_lda_documents.csv"

    print(f"Writing missing document table to {missing_csv_path}")
    write_missing_documents_csv(missing_csv_path, missing_rows)

    print(
        "Found "
        f"{len(missing_rows)} missing documents out of "
        f"{len(dataset_documents)} dataset files."
    )


if __name__ == "__main__":
    main()
