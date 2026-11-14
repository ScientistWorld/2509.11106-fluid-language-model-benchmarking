#!/bin/bash
# CPU job: hardening audit. Runs the reference, a better and a weaker variant and
# every shortcut probe through the unchanged evaluate_{test,train}.sh, leaving the
# reference method's scores in scoring/, then analyses the audit cohort.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONDONTWRITEBYTECODE=1
echo "Start: $(date -u)"
python3 -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"
id
bash hardening/run_audit.sh
python3 hardening/analyze.py
python3 validate.py --post-design --compare || true
echo "End: $(date -u)"
