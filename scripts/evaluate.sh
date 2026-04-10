#!/bin/bash
# Evaluate a FLUID BENCHMARKING run and write scoring/scores.json.
#
# This script is self-contained and uses only code in eval/.

set -euo pipefail

cd /home/user
export PYTHONPATH="/home/user"

RESULTS_FILE="${1:-/home/user/checkpoints/experiment_results.json}"

if [ ! -f "$RESULTS_FILE" ]; then
    echo "Error: results file not found: $RESULTS_FILE"
    exit 1
fi

python3 -m eval.score_results --results "$RESULTS_FILE" --output /home/user/scoring/scores.json
python3 /home/user/validate.py --compare
