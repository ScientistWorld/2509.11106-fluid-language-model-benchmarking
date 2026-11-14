#!/bin/bash
# CPU job entrypoint: reproduce the reference end to end.
#
# Runs the reference method (scripts/method.sh = paper's FLUID BENCHMARKING)
# through both evaluators and validates the workspace. The hardening audit that
# produced HARDENING.md is `bash hardening/run_audit.sh && python3 hardening/analyze.py`.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONDONTWRITEBYTECODE=1
echo "Start: $(date -u)"
bash scripts/reproduce.sh
echo "End: $(date -u)"
