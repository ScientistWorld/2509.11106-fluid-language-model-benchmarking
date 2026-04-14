#!/bin/bash
# CPU job entrypoint for the FLUID BENCHMARKING reproduction.
#
# Runs the paper-method implementation on the released item-level data, converts
# the resulting curves into scoring/scores.json, and validates the workspace.

set -euo pipefail

cd /home/user
export PYTHONPATH="/home/user:/home/user/.pkgs"
export TMPDIR="${TMPDIR:-/home/user/tmp}"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR"

echo "=== FLUID BENCHMARKING: Reproduction Job ==="
echo "Start: $(date -u)"
echo ""

bash scripts/reproduce.sh

echo ""
echo "=== scores.json ==="
cat /home/user/scoring/scores.json

echo ""
echo "End: $(date -u)"
