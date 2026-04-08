#!/bin/bash
# Install Python packages for FLUID BENCHMARKING.
#
# The container rootfs available in this environment is extremely small, so
# direct --system installs can fail with ENOSPC. We therefore install into a
# workspace-local site-packages directory and add it via PYTHONPATH in the
# run/eval scripts.

set -euo pipefail

DEPS_DIR="/home/user/.deps/fluid_benchmarking"
mkdir -p "${DEPS_DIR}"

uv pip install \
    --target "${DEPS_DIR}" \
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
