#!/bin/bash
set -e

# ---- USER CONFIG ----
PYTHON_SCRIPT="$(dirname "$(realpath "$0")")/run_convert_maint.py"
PYTHON_EXEC="/usr/global/miniconda/py310_23.1.0-1/envs/workshop_311/bin/python"
LOG_DIR="$HOME/logs"
ACTIVE_EXPERIMENTS_FILE="/data/eeg/scalp/ltp/ACTIVE_EXPERIMENTS.txt"
SCALP_DATA_ROOT="/data/eeg/scalp/ltp"
OUTPUT_ROOT_BASE="/data/LTP_BIDS"
# ---------------------

mkdir -p "$LOG_DIR"

echo "Starting BIDS conversion..."
echo "Date: $(date)"
echo "-----------------------------------"

# Read active experiments (one per line)
if [[ ! -f "$ACTIVE_EXPERIMENTS_FILE" ]]; then
    echo "ERROR: $ACTIVE_EXPERIMENTS_FILE not found"
    exit 1
fi

mapfile -t EXPERIMENTS < <(grep -v '^\s*$' "$ACTIVE_EXPERIMENTS_FILE")

if [[ ${#EXPERIMENTS[@]} -eq 0 ]]; then
    echo "No active experiments found. Exiting."
    exit 0
fi

echo "Active experiments: ${EXPERIMENTS[*]}"

# For each active experiment, convert all of its recently-modified sessions in
# ONE invocation. The Python driver reads recently_modified.json (via
# --recently-modified), filters its job list to exactly those (subject, session)
# pairs, and fans them out across a single Slurm+Dask cluster. Calling it once
# per session (the old behavior) spun up and tore down a whole cluster per
# session — sequential wall-clock with zero parallel benefit.
#
# NOTE on --root: the driver appends "/<experiment>/" to --root itself, so we
# pass the BASE ($OUTPUT_ROOT_BASE). Passing "$OUTPUT_ROOT_BASE/$EXP" here made
# outputs land in the double-nested "$OUTPUT_ROOT_BASE/$EXP/$EXP/".
for EXP in "${EXPERIMENTS[@]}"; do
    RECENT_FILE="$SCALP_DATA_ROOT/$EXP/recently_modified.json"

    if [[ ! -f "$RECENT_FILE" ]]; then
        echo "No recently_modified.json for $EXP — skipping."
        continue
    fi

    echo "Processing $EXP from $RECENT_FILE"
    echo "  -> $EXP (all recently-modified sessions) / output root $OUTPUT_ROOT_BASE/$EXP"
    $PYTHON_EXEC $PYTHON_SCRIPT \
        --experiments "$EXP" \
        --recently-modified "$RECENT_FILE" \
        --root "$OUTPUT_ROOT_BASE" \
        "$@" || echo "  ✗ FAILED: $EXP"
done

echo "-----------------------------------"
echo "Finished at: $(date)"