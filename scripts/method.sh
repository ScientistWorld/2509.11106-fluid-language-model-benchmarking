#!/bin/bash
# The submitted benchmarking method.
#
# This executable IS the submission: scripts/evaluate_train.sh and
# scripts/evaluate_test.sh start it and talk to it over stdin/stdout with the
# line protocol described in briefing/evaluation.md (one session per LM
# checkpoint x benchmark x item budget; at most `budget` item queries per
# session; one score per session). Write logs to stderr only.
#
# The reference submission is the paper's FLUID BENCHMARKING (method/serve.py,
# policy "fluid"). Replace the command below to submit a different method.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR"
exec python3 -m method.serve --policy fluid "$@"
