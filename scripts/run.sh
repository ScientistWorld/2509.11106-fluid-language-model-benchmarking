#!/bin/bash
# Run FLUID BENCHMARKING full evaluation on compute node.
#
# Uses our independent implementation (method/experiment.py) with all
# 6 LMs, all checkpoints, 4 budgets, 3 methods.
#
# PYTHONPATH must include /home/user/.pkgs for numpy/scipy/pandas.

set -euo pipefail

cd /home/user
export PYTHONPATH="/home/user:/home/user/.pkgs"
export TMPDIR="${TMPDIR:-/home/user/tmp}"
mkdir -p "$TMPDIR"

echo "=== FLUID BENCHMARKING: Full Independent Evaluation ==="
echo "Start: $(date -u)"
echo ""

# Verify data exists
if [ ! -f "data/fluid-benchmarking/open_llm_leaderboard_results.json" ]; then
    echo "Data not found. Run scripts/download.sh first."
    exit 1
fi

# Run the full experiment with all 4 budgets and all checkpoints
echo "Step 1: Running FLUID BENCHMARKING experiment (all 6 LMs, all checkpoints, 4 budgets)..."
python3 -m method.experiment \
    --item-budgets 10 50 100 500 \
    --output /home/user/checkpoints/full_experiment_results.json 2>&1

echo ""
echo "Step 2: Running evaluation..."
python3 -m eval.score_results \
    --results /home/user/checkpoints/full_experiment_results.json \
    --output /home/user/scoring/scores.json

echo ""
echo "Step 3: Validating results..."
python3 validate.py --compare 2>&1 || true

echo ""
echo "=== Results ==="
cat /home/user/scoring/scores.json

echo ""
echo "End: $(date -u)"