#!/bin/bash
# Evaluate the visible train slice for a FLUID BENCHMARKING result payload.
#
# Mirrors scripts/evaluate.sh: optional first argument is the pre-generated
# experiment_results.json payload; scores are written under scoring/.

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

python3 -m eval.train.score_slice --results "$RESULTS_FILE" --output "$ROOT_DIR/scoring/scores_train.json"
