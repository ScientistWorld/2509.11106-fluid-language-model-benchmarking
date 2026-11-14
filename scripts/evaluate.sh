#!/bin/bash
# Public development evaluation of the submitted method (scripts/method.sh).
#
# Runs the visible-LM evaluator (identical to scripts/evaluate_train.sh) and
# writes scoring/scores.json. Uses only code in eval/; never imports method/.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONDONTWRITEBYTECODE=1
python3 "$ROOT_DIR/eval/train/score.py" --output "$ROOT_DIR/scoring/scores.json"
