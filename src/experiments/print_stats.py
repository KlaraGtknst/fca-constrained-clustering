import csv
import json
from pathlib import Path


def compute_and_write_stats(input_path: Path, output_path: Path) -> None:
    with input_path.open() as f:
        context = json.load(f)

    columns = context.get("columns", [])
    rows = context.get("data", [])

    num_topics = len(columns)
    num_docs = len(rows)

    col_sums = [0] * num_topics
    total_ones = 0

    for row in rows:
        row_sum = 0
        for idx, val in enumerate(row):
            if val:
                col_sums[idx] += 1
                row_sum += val
        total_ones += row_sum

    if num_topics:
        avg_docs_per_topic = total_ones / num_topics
        min_docs_per_topic = min(col_sums)
        max_docs_per_topic = max(col_sums)
    else:
        avg_docs_per_topic = 0.0
        min_docs_per_topic = 0
        max_docs_per_topic = 0

    avg_topics_per_doc = total_ones / num_docs if num_docs else 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "total_topics",
                "avg_docs_per_topic",
                "min_docs_per_topic",
                "max_docs_per_topic",
                "avg_topics_per_doc",
            ]
        )
        writer.writerow(
            [
                num_topics,
                avg_docs_per_topic,
                min_docs_per_topic,
                max_docs_per_topic,
                avg_topics_per_doc,
            ]
        )


def main() -> None:
    input_path = Path("resources/banksearch/fca_gt_context.json")
    output_path = Path("resources/banksearch/fca_gt_context_stats.csv")
    compute_and_write_stats(input_path, output_path)


if __name__ == "__main__":
    main()
