#!/bin/bash
# Install Python packages for FLUID BENCHMARKING.
#
set -euo pipefail

python3 -m pip install --user \
    numpy \
    scipy \
    pandas
