#!/bin/bash
# Run the matched subset baselines from the paper's Table 2 comparison.
#
# The experiment runner computes RANDOM, RANDOM_IRT, and FLUID together because
# they share the same released response matrices, item budgets, and random item
# samples. Use scripts/evaluate.sh on the resulting JSON to score the baselines.

set -euo pipefail

cd /home/user
export PYTHONPATH="/home/user:/home/user/.pkgs"

python3 -m method.experiment --item-budgets 10 50 100 500 "$@"
