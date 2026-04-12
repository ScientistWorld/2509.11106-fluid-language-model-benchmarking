#!/bin/bash
# Run the full FLUID BENCHMARKING experiment end-to-end.
#
# This runs our independent implementation (method/experiment.py) which:
# 1. Loads pre-fitted IRT models and LM checkpoint evaluation data
# 2. Computes ability estimates for RANDOM, RANDOM_IRT, and FLUID methods
# 3. Computes validity (rank distance), variance (total variation), and saturation metrics
# 4. Writes results to scoring/scores.json

set -euo pipefail

cd /home/user
export PYTHONPATH="/home/user:/home/user/.deps/fluid_benchmarking"

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
python3 -m method.experiment --item-budgets 10 50 100 500 --output /home/user/checkpoints/full_experiment_results.json

echo ""
echo "Step 2: Running evaluation..."
python3 -m eval.score_results --results /home/user/checkpoints/full_experiment_results.json --output /home/user/scoring/scores.json

echo ""
echo "Step 3: Validating results..."
python3 validate.py --compare || true

echo ""
echo "=== Results ==="
cat /home/user/scoring/scores.json

echo ""
echo "End: $(date -u)"
