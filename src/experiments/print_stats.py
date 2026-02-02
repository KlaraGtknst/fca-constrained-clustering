import csv
import json
from pathlib import Path

from matplotlib import pyplot as plt
import pandas as pd


def compute_and_write_stats(input_path: Path, output_path: Path) -> pd.DataFrame:
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
        assert row_sum >= 1, f"Row sum should be 1, got {row_sum} for row {row}"
        total_ones += row_sum

    print("Number of documents per topic:", col_sums)
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
    # Build a one-row dataframe for downstream use and the CSV export.
    df = pd.DataFrame(
        [
            {
                "total_topics": num_topics,
                "avg_docs_per_topic": avg_docs_per_topic,
                "min_docs_per_topic": min_docs_per_topic,
                "support_of_min_topic": min_docs_per_topic / num_docs if num_docs else 0.0,
                "max_docs_per_topic": max_docs_per_topic,
                "avg_topics_per_doc": avg_topics_per_doc,
                "num_docs": num_docs,
            }
        ]
    )
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "total_topics",
                "avg_docs_per_topic",
                "min_docs_per_topic",
                "support_of_min_topic",
                "max_docs_per_topic",
                "avg_topics_per_doc",
                "num_docs",
            ]
        )
        writer.writerow(
            [
                df.at[0, "total_topics"],
                df.at[0, "avg_docs_per_topic"],
                df.at[0, "min_docs_per_topic"],
                df.at[0, "support_of_min_topic"],
                df.at[0, "max_docs_per_topic"],
                df.at[0, "avg_topics_per_doc"],
                df.at[0, "num_docs"],
            ]
        )
    return df


def main() -> None:
    print("Computing stats for BankSearch Ground Truth FCA contexts...")
    input_path = Path("resources/banksearch/ground_truth/fca_gt_context.json")
    output_path = Path("resources/banksearch/ground_truth/fca_gt_context_stats.csv")
    gt_df = compute_and_write_stats(input_path, output_path)

    print("Computing stats for BankSearch Topic Model FCA contexts...")
    input_path = Path("resources/banksearch/topic_model/fca_topic_model_context.json")
    output_path = Path("resources/banksearch/topic_model/fca_topic_model_context_stats.csv")
    tm_df = compute_and_write_stats(input_path, output_path)

    # combine both dataframes for easier comparison
    gt_df["type"] = "ground_truth"
    tm_df["type"] = "topic_model"
    combined_df = pd.concat(
        [gt_df, tm_df], axis=0
    )
    combined_output_path = Path("resources/banksearch/fca_contexts_comparison_stats.csv")
    combined_output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(combined_output_path, index=False)
    print(f"Saved combined stats to {combined_output_path}")

    svg_output_path = combined_output_path.with_suffix(".svg")

    def df_to_svg_table(df: pd.DataFrame, out_path: Path, max_rows: int = 50):
        """Convert a dataframe to an SVG table and save to out_path to make it easier to view in README."""
        # Safety: SVG tables get huge fast
        df = df.head(max_rows).copy()

        df = df.round(2)
        df = df.astype(str)

        # Rough sizing: scale figure by rows/cols
        nrows, ncols = df.shape
        fig_w = max(6, min(20, 1.2 * ncols + 2))
        fig_h = max(2, min(30, 0.35 * (nrows + 1) + 1))

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")

        # TODO: round floats to 2 decimal places
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            loc="center",
            cellLoc="left",
            colLoc="left",
        )

        # Improve readability
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.2)

        # Make header a bit bolder
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_text_props(weight="bold")

        fig.tight_layout()
        fig.savefig(out_path, format="svg", bbox_inches="tight")
        plt.close(fig)

    df_to_svg_table(combined_df, svg_output_path)
    print(f"Saved SVG table to {svg_output_path}")


if __name__ == "__main__":
    main()
