#!/bin/bash
# Install Python dependencies for FLUID BENCHMARKING.
# Uses uv for faster installation.

set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    python3 -m pip install --break-system-packages uv
fi

env \
    -u SSL_CERT_FILE \
    -u SSL_CERT_DIR \
    UV_SYSTEM_PYTHON=1 \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    UV_NO_PROGRESS=1 \
    uv pip install --system numpy scipy pandas
