#!/bin/bash
# Visible train evaluation. No arguments.
#
# Same protocol, metrics and failure rules as the hidden test, on the visible
# LMs (Amber-6.7B, OLMo1-7B, OLMo2-7B). Writes scoring/scores_train.json.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONDONTWRITEBYTECODE=1
python3 "$ROOT_DIR/eval/train/score.py" --output "$ROOT_DIR/scoring/scores_train.json"
