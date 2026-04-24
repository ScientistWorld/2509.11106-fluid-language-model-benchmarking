#!/bin/bash
# CPU job entrypoint for the packaged FLUID BENCHMARKING reproduction.
#
# Runs the canonical end-to-end script inside the compute container and prints
# the resulting scores file for quick inspection in the job log.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR"
export TMPDIR="${TMPDIR:-$ROOT_DIR/tmp}"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR"

echo "=== FLUID BENCHMARKING: Reproduction Validation Job ==="
echo "Start: $(date -u)"
echo ""

bash scripts/reproduce.sh

python3 - <<'PY'
import json
from pathlib import Path

scores_path = Path.cwd() / "scoring" / "scores.json"
scores = json.loads(scores_path.read_text())
print("\n=== scores.json ===")
print(json.dumps(scores, indent=2))
PY

echo ""
echo "End: $(date -u)"
