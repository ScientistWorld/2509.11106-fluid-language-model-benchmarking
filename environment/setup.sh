#!/bin/bash
# Packages (numpy, scipy, pandas) are installed in container.def.
# This script verifies they're available.

set -euo pipefail

python3 -c "import numpy, scipy, pandas; print('Packages OK')"
