#!/bin/bash
# Install Python dependencies for FLUID BENCHMARKING.
# Uses uv for faster installation.

set -euo pipefail

uv pip install --system numpy scipy pandas
