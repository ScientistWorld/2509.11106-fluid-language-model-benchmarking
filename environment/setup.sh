#!/bin/bash
# Python dependencies for the FLUID BENCHMARKING reproduction.
#
# The full method/evaluation path imports numpy, pandas, and scipy. Keep these
# installs here rather than in runtime scripts so a fresh container rebuild has
# the same environment as the validated jobs.

set -euo pipefail

uv pip install --system \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scipy==1.13.1
