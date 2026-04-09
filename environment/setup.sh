#!/bin/bash
# Install Python packages for FLUID BENCHMARKING.
#
set -euo pipefail

uv pip install \
    --system \
    numpy \
    scipy \
    pandas
