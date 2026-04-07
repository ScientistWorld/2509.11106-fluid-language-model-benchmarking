#!/bin/bash
# Install Python packages into the container overlay.
#
# This script is run by the system on the login node (which has internet)
# inside the container with a persistent overlay. Changes persist across
# job submissions without rebuilding the container.
#
# Put ALL pip installs here, NOT in container.def.
# Only use container.def for the base Docker image and apt-get/yum packages.
#
# IMPORTANT: Install into the SYSTEM Python, not a virtual environment.
# The overlay may be recreated on container rebuilds — a venv inside
# it will be lost and all subsequent jobs will fail.
# Use: python3 -m pip install <packages>
# Do NOT create a venv (python -m venv, virtualenv, etc.).

set -e

# Core numerical libraries needed for IRT computations and metrics
python3 -m pip install --no-cache-dir \
    numpy \
    scipy \
    scikit-learn \
    python-dateutil \
    pytz \
    pyyaml \
    filelock \
    pyarrow \
    pandas \
    multiprocess \
    xxhash \
    2>&1 | tail -5

# HuggingFace packages for data download (only needed on login node)
# not strictly needed in container since data is pre-downloaded in workspace
python3 -m pip install --no-cache-dir \
    huggingface-hub \
    datasets \
    2>&1 | tail -5