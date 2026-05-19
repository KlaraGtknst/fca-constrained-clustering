import json
import logging
import os
from pathlib import Path
import subprocess
from typing import Dict, Any
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.plot_concept_lattice import IcebergLatticePlotter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


RESULTS_DIR = PROJECT_ROOT / "results" / "context_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def call_clojure_analyze(cxt_path: str, min_support: float = 0.0) -> Dict[str, Any]:
    clj = [
        "clojure",
        "-M",
        "-e",
        ' (load-file "src/context_comparison/context_comparison.clj")'
        f' (clojure.core/println (user/analyze-context "{cxt_path}" {min_support}))',
    ]
    cwd_candidates = [
        str(PROJECT_ROOT),
        str(Path.home() / "Developer" / "Uni" / "FCA" / "fca-constrained-clustering"),
    ]
    cwd = next((p for p in cwd_candidates if os.path.exists(p)), str(PROJECT_ROOT))
    logger.info(f"Calling Clojure analyze on {cxt_path} (cwd={cwd})...")
    out = subprocess.check_output(clj, text=True, cwd=cwd, stderr=subprocess.STDOUT)
    logger.info(out)
    # The function already writes JSON stats; optionally parse last JSON-looking line if needed.
    return {}


if __name__ == "__main__":
    # Paths provided in the request
    min_supp = 0.05
    iceberg_filename = f"banksearch_{min_supp}_iceberg"
    min_support_iceberg_cxt1 = str(PROJECT_ROOT / "resources" / "banksearch" / "topic_model" / f"{iceberg_filename}.cxt")
    mlb_expanded_cxt2 = str(PROJECT_ROOT / "resources" / "banksearch" / "ground_truth" / "mlb_expanded.cxt")
    assert Path(min_support_iceberg_cxt1).exists(), f"Context file not found: {min_support_iceberg_cxt1}"
    assert Path(mlb_expanded_cxt2).exists(), f"Context file not found: {mlb_expanded_cxt2}"

    # Analyze both (full lattice by default)
    call_clojure_analyze(min_support_iceberg_cxt1, 0.0)
    call_clojure_analyze(mlb_expanded_cxt2, 0.0)

    # Plot PNGs from the EDN concepts that Clojure produced
    # FIXME: use Burmeister format to save formal (iceberg) context
    edn1_path = Path(PROJECT_ROOT) / f"resources/banksearch/topic_model/{iceberg_filename}.edn"
    edn2_path = RESULTS_DIR / "mlb_expanded_concepts.edn"

    img1_path = RESULTS_DIR / f"{iceberg_filename}_lattice.svg"
    img2_path = RESULTS_DIR / "mlb_expanded_lattice.svg"

    if edn1_path.exists():
        IcebergLatticePlotter().plot(edn1_path, img1_path, min_support=None)
        logger.info(f"Saved PNG to {img1_path}")
    else:
        logger.warning(f"EDN not found: {edn1_path}")

    if edn2_path.exists():
        IcebergLatticePlotter().plot(edn2_path, img2_path, min_support=None, omit_transitive_intents=True)
        logger.info(f"Saved PNG to {img2_path}")
    else:
        logger.warning(f"EDN not found: {edn2_path}")

    # Optionally, aggregate stats that the Clojure side already saved
    stats1_path = RESULTS_DIR / f"{iceberg_filename}_stats.json"
    stats2_path = RESULTS_DIR / "mlb_expanded_stats.json"

    comparison_path = RESULTS_DIR / "comparison_summary.json"
    summary: Dict[str, Any] = {}
    if stats1_path.exists():
        with open(stats1_path) as f:
            summary["iceberg_context"] = json.load(f)
    if stats2_path.exists():
        with open(stats2_path) as f:
            summary["mlb_expanded"] = json.load(f)

    if summary:
        with open(comparison_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Wrote comparison summary to {comparison_path}")
    else:
        logger.warning("No stats JSON files found to summarize.")
