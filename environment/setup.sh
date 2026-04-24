#!/bin/bash
# Install Python dependencies for FLUID BENCHMARKING.
# Uses uv for faster installation.

set -euo pipefail

# The scheduler injects cert paths that do not exist inside this container.
# Unset them so uv does not emit warning-only stderr on successful installs.
if [ -n "${SSL_CERT_FILE:-}" ] && [ ! -f "${SSL_CERT_FILE}" ]; then
    unset SSL_CERT_FILE
fi
if [ -n "${SSL_CERT_DIR:-}" ] && [ ! -d "${SSL_CERT_DIR}" ]; then
    unset SSL_CERT_DIR
fi

export UV_SYSTEM_PYTHON=1
export UV_BREAK_SYSTEM_PACKAGES=1
export UV_NO_PROGRESS=1

if ! command -v uv >/dev/null 2>&1; then
    python3 -m pip install --break-system-packages uv
fi

uv pip install --system numpy scipy pandas
