#!/bin/bash

# Exit immediately if a command fails
set -e

# ---- USER CONFIG ----
PYTHON_SCRIPT="convert_to_bids.py"
PYTHON_EXEC="/usr/global/miniconda/py310_23.1.0-1/envs/workshop_311"   # or your conda env python
LOG_DIR="$HOME/logs"
# ---------------------

mkdir -p "$LOG_DIR"

echo "Starting BIDS conversion..."
echo "Date: $(date)"
echo "Arguments: $@"
echo "-----------------------------------"

$PYTHON_EXEC $PYTHON_SCRIPT "$@"

echo "-----------------------------------"
echo "Finished at: $(date)"