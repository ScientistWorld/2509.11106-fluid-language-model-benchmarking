#!/bin/bash
# Install the lightweight numerical stack required by the FLUID reproduction.

set -euo pipefail

uv pip install --system \
    numpy \
    pandas \
    scipy
