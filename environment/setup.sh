#!/bin/bash
# Install Python dependencies for FLUID BENCHMARKING.
# Uses uv for faster installation.

set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    python3 -m pip install --break-system-packages uv
fi

uv pip install --system numpy scipy pandas
