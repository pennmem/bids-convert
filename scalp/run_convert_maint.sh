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

# For each active experiment, check for recently_modified.json
for EXP in "${EXPERIMENTS[@]}"; do
    RECENT_FILE="$SCALP_DATA_ROOT/$EXP/recently_modified.json"

    if [[ ! -f "$RECENT_FILE" ]]; then
        echo "No recently_modified.json for $EXP — skipping."
        continue
    fi

    echo "Processing $EXP from $RECENT_FILE"

    # Parse JSON: extract each subject and its sessions, then call the python script per session
    python3 -c "
import json, sys
with open('$RECENT_FILE') as f:
    data = json.load(f)
for subj, sessions in data.items():
    for ses in sessions:
        print(f'{subj} {ses}')
" | while read -r SUBJECT SESSION; do
        echo "  -> $EXP / $SUBJECT / session $SESSION / output root $OUTPUT_ROOT_BASE/$EXP"
        $PYTHON_EXEC $PYTHON_SCRIPT \
            --experiments "$EXP" \
            --subject "$SUBJECT" \
            --session "$SESSION" \
            --root "$OUTPUT_ROOT_BASE/$EXP" \
            "$@" || echo "  ✗ FAILED: $SUBJECT $EXP $SESSION"
    done
done

echo "-----------------------------------"
echo "Finished at: $(date)"