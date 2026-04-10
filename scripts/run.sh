#!/bin/bash
# Full FLUID BENCHMARKING reproduction run.
#
# Runs the paper's three ablation methods (RANDOM, RANDOM IRT, FLUID)
# at item budgets 10/50/100/500 on all 6 LMs × 6 benchmarks with all checkpoints,
# then converts outputs into scoring/scores.json.

set -euo pipefail

cd /home/user
export PYTHONPATH="/home/user"

echo "=== FLUID BENCHMARKING: Full Reproduction ==="
echo "Start: $(date -u)"

python3 -m method.experiment --item-budgets 10 50 100 500 --seed 42

echo ""
echo "Converting results..."
python3 -m eval.score_results --results /home/user/checkpoints/experiment_results.json --output /home/user/scoring/scores.json

echo ""
echo "Comparing against paper reference..."
python3 validate.py --compare || true

echo ""
echo "End: $(date -u)"
