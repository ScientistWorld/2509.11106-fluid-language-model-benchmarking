#!/bin/bash
# CPU job entrypoint for the FLUID BENCHMARKING split evaluation.
#
# Scores the existing final results payload against the visible train slice and
# held-out test slice, then checks that both files cover the same methods.

set -euo pipefail

cd /home/user
export PYTHONPATH="/home/user:/home/user/.pkgs"
export TMPDIR="${TMPDIR:-/home/user/tmp}"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR"

echo "=== FLUID BENCHMARKING: Split Evaluation Job ==="
echo "Start: $(date -u)"
echo ""

bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh

python3 - <<'PY'
import json
from pathlib import Path

train_path = Path("/home/user/scoring/scores_train.json")
test_path = Path("/home/user/scoring/scores_test.json")

train_scores = json.loads(train_path.read_text())
test_scores = json.loads(test_path.read_text())

for path, scores in ((train_path, train_scores), (test_path, test_scores)):
    print(f"\n=== {path.name} ===")
    print(json.dumps(scores, indent=2))

train_experiments = train_scores["experiments"]
test_experiments = test_scores["experiments"]
for experiment_name in sorted(set(train_experiments) | set(test_experiments)):
    train_methods = sorted(train_experiments.get(experiment_name, {}).get("results", {}))
    test_methods = sorted(test_experiments.get(experiment_name, {}).get("results", {}))
    if train_methods != test_methods:
        raise SystemExit(
            f"Method mismatch for {experiment_name}: "
            f"train={train_methods}, test={test_methods}"
        )

print("\nTrain/test method identifiers match for every experiment.")
PY

echo ""
echo "End: $(date -u)"
