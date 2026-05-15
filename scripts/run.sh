#!/bin/bash
# CPU job entrypoint for the real train/test FLUID BENCHMARKING evaluation run.
#
# Scores the frozen experiment payload into scoring/scores_{train,test}.json
# and prints a small integrity summary for the job log.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR"
export TMPDIR="${TMPDIR:-$ROOT_DIR/tmp}"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR"

echo "=== FLUID BENCHMARKING: Train/Test Evaluation Job ==="
echo "Start: $(date -u)"
echo ""

bash scripts/evaluate_train.sh "$ROOT_DIR/checkpoints/experiment_results.json"
bash scripts/evaluate_test.sh "$ROOT_DIR/checkpoints/experiment_results.json"

python3 - <<'PY'
import json
from pathlib import Path

root = Path.cwd()
train_path = root / "scoring" / "scores_train.json"
test_path = root / "scoring" / "scores_test.json"
train = json.loads(train_path.read_text())
test = json.loads(test_path.read_text())

print("\n=== scores_train.json ===")
print(json.dumps(train, indent=2))
print("\n=== scores_test.json ===")
print(json.dumps(test, indent=2))

print("\n=== integrity summary ===")
for experiment_name, train_experiment in train["experiments"].items():
    train_keys = sorted(train_experiment["results"])
    test_keys = sorted(test["experiments"][experiment_name]["results"])
    print(
        f"{experiment_name}: train_results={len(train_keys)} "
        f"test_results={len(test_keys)} keys_match={train_keys == test_keys}"
    )
PY

echo ""
echo "End: $(date -u)"
