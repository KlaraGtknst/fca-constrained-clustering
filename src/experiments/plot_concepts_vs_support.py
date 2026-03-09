import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

from edn_format import loads
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _contains_bool(value):
    if isinstance(value, bool):
        return True
    if isinstance(value, list):
        return any(_contains_bool(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_bool(item) for item in value.values())
    return False


def _convert_bools_to_ints(value):
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, list):
        return [_convert_bools_to_ints(item) for item in value]
    if isinstance(value, dict):
        return {key: _convert_bools_to_ints(item) for key, item in value.items()}
    return value


def ensure_zero_one_json(source_path):
    translated_path = source_path.replace(".json", "_01.json")
    with open(source_path, "r", encoding="utf-8") as source_file:
        data = json.load(source_file)

    if not _contains_bool(data):
        return source_path

    if os.path.exists(translated_path):
        with open(translated_path, "r", encoding="utf-8") as translated_file:
            translated_data = json.load(translated_file)
        if not _contains_bool(translated_data):
            return translated_path

    converted = _convert_bools_to_ints(data)
    with open(translated_path, "w", encoding="utf-8") as translated_file:
        json.dump(converted, translated_file, indent=2, ensure_ascii=True)
    return translated_path


def read_edn_concepts_count(path: str) -> int:
    with open(path, "r") as f:
        edn_data = loads(f.read())
    return len(edn_data)


def read_context_stats(path: str) -> tuple[int, int]:
    with open(path, "r", encoding="utf-8") as source_file:
        data = json.load(source_file)
    columns = data.get("columns", [])
    rows = data.get("data", [])
    return len(rows), len(columns)


def run_iceberg(context_path: str, min_support: float, edn_path: str, repo_root: Path) -> None:
    cmd = [
        "clojure",
        "-M",
        "-e",
        '(load-file "src/experiments/iceberg_lattice.clj") '
        f'(run-iceberg "{context_path}" {min_support} "{edn_path}")',
    ]
    subprocess.check_output(
        cmd,
        text=True,
        cwd=str(repo_root),
        stderr=subprocess.STDOUT,
    )


def build_supports(step: float):
    steps = int(round(1.0 / step))
    supports = [round(i * step, 10) for i in range(steps + 1)]
    supports[-1] = 1.0
    return supports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot number of iceberg concepts vs minimum support."
    )
    parser.add_argument(
        "--context",
        default="/resources/banksearch/topic_model/fca_topic_model_context.json",
        help="Path to the JSON context file.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Step size for min_support from 0.0 to 1.0.",
    )
    parser.add_argument(
        "--output",
        default="resources/banksearch/topic_model/plots/concepts_vs_support.svg",
        help="Output SVG path.",
    )
    args = parser.parse_args()

    if args.step <= 0 or args.step > 1:
        raise ValueError("step must be in (0, 1].")

    repo_root = Path(__file__).resolve().parents[2]
    context_path = ensure_zero_one_json(str(repo_root) + args.context)

    supports = build_supports(args.step)
    counts = []
    edn_dir = repo_root / "resources/banksearch/topic_model/iceberg_sweep"
    edn_dir.mkdir(parents=True, exist_ok=True)

    for support in supports:
        support_str = f"{support:.2f}"
        logger.info("min_support=%s", support_str)
        cxt_path = edn_dir / f"banksearch_{support_str}_iceberg"
        if not cxt_path.exists():
            run_iceberg(context_path, support, str(cxt_path), repo_root)
        counts.append(read_edn_concepts_count(str(cxt_path)+".edn"))
        logger.info("min_support=%s concepts=%s", support_str, counts[-1])

    num_docs, num_topics = read_context_stats(context_path)
    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_count = max(counts) if counts else 0

    if max_count > 500:
        fig, (ax_high, ax_low) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(5, 5),
            gridspec_kw={"height_ratios": [1, 2], "hspace": 0.1},
        )
        ax_high.plot(supports, counts, marker="o", linewidth=1)
        ax_low.plot(supports, counts, marker="o", linewidth=1)

        ax_low.set_ylim(0, 100)
        ax_high.set_ylim(550, max_count * 1.02)

        ax_high.spines["bottom"].set_visible(False)
        ax_low.spines["top"].set_visible(False)
        ax_high.tick_params(labeltop=False,axis="both", labelsize=10)
        ax_low.xaxis.tick_bottom()

        d = 0.015
        kwargs = dict(transform=ax_high.transAxes, color="k", clip_on=False, linewidth=1)
        ax_high.plot((-d, +d), (-d, +d), **kwargs)
        ax_high.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs.update(transform=ax_low.transAxes)
        ax_low.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax_low.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

        ax_low.set_xlabel("Minimum support", fontsize=12)
        ax_low.set_ylabel("Number of concepts", fontsize=12)
        for ax in (ax_high, ax_low):
            ax.xaxis.set_major_locator(MultipleLocator(0.1))
            ax.yaxis.set_major_locator(MultipleLocator(10))
            ax.grid(True, alpha=0.3)
        ax_low.set_xlim(0, 1)
    else:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(supports, counts, marker="o", linewidth=1)
        ax.set_xlabel("Minimum support", fontsize=12)
        ax.set_ylabel("Number of concepts", fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.grid(True, alpha=0.3)

    fig.suptitle("Effect of Minimum Support on Iceberg Concepts", y=0.97, fontsize=13)
    fig.text(
        0.5,
        0.92,
        f"# Documents = # Objects = {num_docs} | # Topics = # Attributes = {num_topics}",
        ha="center",
        va="top",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    logger.info("Saved plot to %s", output_path)


if __name__ == "__main__":
    main()
