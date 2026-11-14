#!/bin/bash
# Hidden test evaluation. No arguments.
#
# Runs the submitted method (scripts/method.sh) through the item-query protocol
# on the held-out LMs whose responses live only in eval/test/, recomputes every
# metric from the method's raw per-session scores, and writes
# scoring/scores_test.json. A failed run is written with the worst value of
# every scored metric.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONDONTWRITEBYTECODE=1
python3 "$ROOT_DIR/eval/test/score.py" --output "$ROOT_DIR/scoring/scores_test.json"
