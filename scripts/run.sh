#!/bin/bash
# Run the FLUID BENCHMARKING evaluation end-to-end.
#
# This script:
# 1. Loads pre-computed HuggingFace upstream data
# 2. Computes validity, variance, and saturation metrics
# 3. Writes results to scoring/scores.json
#
# This is a CPU-only workload (no GPU needed).

set -euo pipefail

cd /home/user
export PYTHONPATH="/home/user:/home/user/.pkgs"
export TMPDIR="${TMPDIR:-/home/user/tmp}"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR"

echo "=== FLUID BENCHMARKING: Full Independent Evaluation ==="
echo "Start: $(date -u)"
echo ""

# Verify upstream data exists
if [ ! -f "/home/user/tmp_fluid_upstream/results/experiments.jsonl" ]; then
    echo "Upstream data not found at /home/user/tmp_fluid_upstream/results/experiments.jsonl"
    exit 1
fi

# Run the evaluation script (computes all metrics + writes scores.json)
python3 -m eval.run_evaluation 2>&1

echo ""
echo "Step 2: Validating results..."
python3 validate_new.py --compare 2>&1 || true

echo ""
echo "=== Results ==="
cat /home/user/scoring/scores.json

echo ""
echo "End: $(date -u)"