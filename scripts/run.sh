#!/bin/bash
# Compute FLUID BENCHMARKING metrics from pre-computed upstream results.

set -euo pipefail

cd /home/user
export PYTHONPATH="/home/user:/home/user/.deps/fluid_benchmarking"

echo "=== Computing FLUID BENCHMARKING metrics ==="
echo "Start: $(date -u)"

python3 /home/user/eval/compute_from_upstream.py

echo ""
echo "Comparing against paper reference..."
python3 /home/user/validate.py --compare || true

echo ""
echo "End: $(date -u)"
