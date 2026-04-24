#!/bin/bash
# Run the paper's proposed experiment on the released data.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR"
python3 -m method.experiment --item-budgets 10 50 100 500 "$@"
