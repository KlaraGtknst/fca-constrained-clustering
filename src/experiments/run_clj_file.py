import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import networkx as nx
from matplotlib import pyplot as plt
from edn_format import loads

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
    try:
        with open(source_path, "r", encoding="utf-8") as source_file:
            data = json.load(source_file)
    except FileNotFoundError:
        print(f"File {source_path} not found. Current pwd: {os.getcwd()}")
        raise (FileNotFoundError)

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
context_path = ensure_zero_one_json(
        str(PROJECT_ROOT) + "/resources/banksearch/topic_model/fca_topic_model_context.json"
)

min_support = 0.05
cxt_path = f"resources/banksearch/topic_model/banksearch_{min_support}_iceberg"
svg_path = (
    f"resources/banksearch/topic_model/plots/banksearch_{min_support}_iceberg.svg"
)
iceberg_context_csv_path = "resources/banksearch/topic_model/iceberg_context.csv"
cmd = [
    "clojure",
    "-M",
    "-e",
    '(load-file "src/experiments/iceberg_lattice.clj") '
    f'(run-iceberg "{context_path}" {min_support} "{cxt_path}")',
]

try:
    path = "/Users/klara/Developer/fca-constrained-clustering"
    if not os.path.exists(path):
        path = "/Users/klara/Developer/Uni/FCA/fca-constrained-clustering"
    out = subprocess.check_output(
        cmd,
        text=True,
        cwd=path,
        stderr=subprocess.STDOUT,
    )
    logger.info(out)
except subprocess.CalledProcessError as e:
    logger.error(e.output)
