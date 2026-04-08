#!/bin/bash
# Run the paper's proposed experiment on the released data.

set -euo pipefail

cd /home/user
python3 -m method.experiment --item-budgets 10 50 100 500 "$@"
