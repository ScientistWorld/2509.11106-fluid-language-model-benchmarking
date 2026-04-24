#!/bin/bash
# Install the lightweight numerical stack required by the FLUID reproduction.
# Re-touched after removing the old /home/user/.pkgs no-op path so the
# scheduler performs a fresh setup run against the current environment state.

set -euo pipefail

uv pip install --system \
    numpy \
    pandas \
    scipy
