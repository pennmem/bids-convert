#!/bin/bash
set -e

# ---- USER CONFIG ----
PYTHON_SCRIPT="$(dirname "$(dirname "$(realpath "$0")")")/bids_convert.py"
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
# ONE invocation. The driver reads recently_modified.json (via
# --recently-modified), filters its job list to exactly those (subject, session)
# pairs, and fans them out across a single Slurm+Dask cluster. Calling it once
# per session (the old behavior) spun up and tore down a whole cluster per
# session — sequential wall-clock with zero parallel benefit.
#
# --root is used verbatim by the driver, so the per-experiment dataset path is
# spelled out here: /data/LTP_BIDS/<Experiment>/sub-*/...
FAILED=()
for EXP in "${EXPERIMENTS[@]}"; do
    RECENT_FILE="$SCALP_DATA_ROOT/$EXP/recently_modified.json"

    if [[ ! -f "$RECENT_FILE" ]]; then
        echo "No recently_modified.json for $EXP — skipping."
        continue
    fi

    echo "Processing $EXP from $RECENT_FILE"
    echo "  -> $EXP (all recently-modified sessions) / output root $OUTPUT_ROOT_BASE/$EXP"
    if ! $PYTHON_EXEC $PYTHON_SCRIPT \
        --modality scalp \
        --experiments "$EXP" \
        --recently-modified "$RECENT_FILE" \
        --root "$OUTPUT_ROOT_BASE/$EXP" \
        "$@"; then
        echo "  ✗ FAILED: $EXP"
        FAILED+=("$EXP")
    fi
done

echo "-----------------------------------"
echo "Finished at: $(date)"

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Experiments with conversion failures: ${FAILED[*]}"
    exit 1
fi