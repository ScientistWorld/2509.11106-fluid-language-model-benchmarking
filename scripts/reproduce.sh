#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# This script runs inside the compute container and executes the released-data
# reproduction end to end. The method runner generates the raw experiment
# payload and the evaluation runner converts it into scoring/scores.json.
#
# Assumes:
#   - Container is built (environment/container.sif)
#   - Data and models are downloaded (via scripts/download.sh)
#   - Working directory is /home/user
#
# This is the "gold standard" reproduction —
# You do not need to run this, but keep this in good shape
# so other agents can easily run a full loop of reproduction and get the paper's numbers.

set -e

cd /home/user
export PYTHONPATH="/home/user:/home/user/.pkgs"

echo "=== FLUID BENCHMARKING: Full Reproduction ==="
echo ""

# Verify data exists
if [ ! -f "data/fluid-benchmarking/open_llm_leaderboard_results.json" ]; then
    echo "Data not found. Run scripts/download.sh first."
    exit 1
fi

echo "Step 1: Running FLUID BENCHMARKING method..."
bash scripts/method.sh

echo ""
echo "Step 2: Running evaluation..."
bash scripts/evaluate.sh /home/user/checkpoints/experiment_results.json

echo ""
echo "Step 3: Validating results..."
python3 validate.py --compare 2>&1 || true

echo ""
echo "=== Reproduction complete ==="
echo "Results are in scoring/scores.json"
echo "Detailed results in checkpoints/experiment_results.json"
