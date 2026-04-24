#!/bin/bash
# Install Python dependencies for FLUID BENCHMARKING.

set -euo pipefail

python3 -m pip install --break-system-packages --no-cache-dir \
    numpy \
    scipy \
    pandas
