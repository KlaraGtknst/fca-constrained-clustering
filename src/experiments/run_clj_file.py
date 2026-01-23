import json
import os
import subprocess


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

context_path = ensure_zero_one_json(
    "resources/banksearch/fca_topic_model_context.json"
)

cmd = [
    "clojure",
    "-M",
    "-e",
    '(load-file "src/experiments/iceberg_lattice.clj") '
    f'(run-iceberg "{context_path}" 0.9 "resources/banksearch/my_iceberg.edn")',
]

try:
    out = subprocess.check_output(
        cmd,
        text=True,
        cwd="/Users/klara/Developer/fca-constrained-clustering",
        stderr=subprocess.STDOUT,
    )
    print(out, end="")
except subprocess.CalledProcessError as e:
    print(e.output)
