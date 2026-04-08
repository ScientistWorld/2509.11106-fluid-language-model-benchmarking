#!/bin/bash
# CPU-only reproduction run for FLUID BENCHMARKING.
#
# This evaluates RANDOM, RANDOM IRT, and FLUID on the released benchmark data
# at item budgets 10/50/100/500, then writes scoring/scores.json.

set -euo pipefail

cd /home/user

echo "=== FLUID BENCHMARKING reproduction ==="
echo "Start: $(date -u)"

python3 -m method.experiment --item-budgets 10 50 100 500
bash /home/user/scripts/evaluate.sh /home/user/checkpoints/experiment_results.json

echo "End: $(date -u)"
