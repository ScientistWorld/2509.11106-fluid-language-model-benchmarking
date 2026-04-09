#!/bin/bash
# Install Python packages for FLUID BENCHMARKING.
#
set -euo pipefail

uv pip install \
    --system \
    numpy \
    scipy \
    pandas \
    scikit-learn \
    pyyaml \
    python-dateutil \
    pytz \
    tqdm \
    huggingface-hub \
    datasets
