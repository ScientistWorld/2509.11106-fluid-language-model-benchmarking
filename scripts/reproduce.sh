#!/bin/bash
# Reproduce the reference results end to end (runs inside the compute container).
#
# The reference method (paper's FLUID BENCHMARKING, scripts/method.sh) is run by
# the evaluators through the item-query protocol:
#   1. visible LMs  -> scoring/scores_train.json   (scripts/evaluate_train.sh)
#   2. visible LMs  -> scoring/scores.json         (scripts/evaluate.sh, public dev score)
#   3. held-out LMs -> scoring/scores_test.json    (scripts/evaluate_test.sh)
#
# Assumes scripts/download.sh has been run (data/downloads/ and eval/*/data/).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -f data/downloads/fluid-benchmarking/open_llm_leaderboard_results.json ]; then
    echo "Data not found. Run scripts/download.sh first."
    exit 1
fi

echo "=== FLUID BENCHMARKING: reference reproduction ($(date -u)) ==="
bash scripts/evaluate_train.sh
bash scripts/evaluate.sh
if [ -f eval/test/data/hidden_responses.json ]; then
    bash scripts/evaluate_test.sh
fi
python3 validate.py --compare
echo "=== Reproduction complete ($(date -u)) ==="
