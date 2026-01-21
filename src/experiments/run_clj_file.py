import subprocess

cmd = [
    "clojure",
    "-e",
    '(load-file "src/experiments/iceberg_lattice.clj") (run-iceberg)'
]

out = subprocess.check_output(
    cmd,
    text=True,
    cwd="/Users/klara/Developer/fca-constrained-clustering",
    stderr=subprocess.STDOUT,
)

print(out, end="")