#!/bin/bash
# Paper ablations served through the same protocol as scripts/method.sh.
#
#   bash scripts/baseline.sh            # RANDOM IRT: random subset + MAP ability
#   bash scripts/baseline.sh random     # RANDOM: random subset + accuracy
#
# The evaluators already compute both ablations themselves (averaged over five
# fixed random subsets) and report them under each experiment's "baselines";
# this script exists so they can also be run as a submitted method, e.g.
#   sed -i 's/--policy fluid/--policy random_irt/' scripts/method.sh

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR"
exec python3 -m method.serve --policy "${1:-random_irt}"
