#!/bin/bash
# Python dependencies are vendored in /home/user/.pkgs and loaded at runtime
# via PYTHONPATH in the run/evaluation scripts. Keep setup as a no-op so
# environment creation does not depend on networked package installation.

set -euo pipefail

echo "Using vendored Python packages from /home/user/.pkgs"
