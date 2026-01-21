import subprocess

cmd = [
    "clojure",
    "-M",
    "-e",
    '(load-file "src/experiments/iceberg_lattice.clj") '
    '(run-iceberg "resources/banksearch/fca_topic_model_context.json" 0.9 "resources/banksearch/my_iceberg.edn")'
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

