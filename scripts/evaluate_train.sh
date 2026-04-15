#!/bin/bash
# Evaluate the visible train slice for a FLUID BENCHMARKING result payload.
#
# Mirrors scripts/evaluate.sh: optional first argument is the pre-generated
# experiment_results.json payload; scores are written under scoring/.

set -euo pipefail

cd /home/user
export PYTHONPATH="/home/user:/home/user/.pkgs"
export TMPDIR="${TMPDIR:-/home/user/tmp}"
mkdir -p "$TMPDIR"

RESULTS_FILE="${1:-/home/user/checkpoints/experiment_results.json}"

if [ ! -f "$RESULTS_FILE" ]; then
    echo "Error: results file not found: $RESULTS_FILE"
    exit 1
fi

python3 /home/user/eval/train/score_slice.py --results "$RESULTS_FILE" --output /home/user/scoring/scores_train.json
