#!/bin/bash
# Install Python dependencies for FLUID BENCHMARKING.

set -euo pipefail

uv pip install --system \
  numpy \
  pandas \
  scipy
