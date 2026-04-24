#!/bin/bash
# Evaluate a FLUID BENCHMARKING run and write scoring/scores.json.
#
# This script is self-contained and uses only code in eval/.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR"
export TMPDIR="${TMPDIR:-$ROOT_DIR/tmp}"
mkdir -p "$TMPDIR"

RESULTS_FILE="${1:-$ROOT_DIR/checkpoints/experiment_results.json}"

if [ ! -f "$RESULTS_FILE" ]; then
    echo "Error: results file not found: $RESULTS_FILE"
    exit 1
fi

python3 -m eval.score_results --results "$RESULTS_FILE" --output "$ROOT_DIR/scoring/scores.json"
python3 "$ROOT_DIR/validate.py" --compare
