#!/usr/bin/env bash
# Translate a test set with an ensemble of N fairseq checkpoints.
#
# Usage:
#   generate_ensemble.sh <config_file> <num_runs>
#
# Arguments:
#   config_file  Path to a dataset config file (see scripts/dataset_configs/).
#                Must define: DATA_BIN, BPE_CODES, TEST_INPUT, CHECKPOINT_DIR,
#                             OUTPUT_DIR, SOURCE_LANG, TARGET_LANG.
#   num_runs     Number of checkpoints to include in the ensemble (runs 0..N-1).
#
# The ensemble hypothesis file is written to OUTPUT_DIR/ensemble_<N>_models/.
# To score the output, use scripts/score.sh.
#
# Example:
#   generate_ensemble.sh scripts/dataset_configs/wmt23_cs_uk.cfg 19

set -euo pipefail

if [[ $# -lt 2 || "$1" == "--help" ]]; then
    sed -n '2,18p' "$0"
    exit 0
fi

source "$1"
NUM_RUNS="$2"

: "${DATA_BIN:?config must define DATA_BIN}"
: "${BPE_CODES:?config must define BPE_CODES}"
: "${TEST_INPUT:?config must define TEST_INPUT}"
: "${CHECKPOINT_DIR:?config must define CHECKPOINT_DIR}"
: "${OUTPUT_DIR:?config must define OUTPUT_DIR}"
: "${SOURCE_LANG:?config must define SOURCE_LANG}"
: "${TARGET_LANG:?config must define TARGET_LANG}"

# Build colon-separated list of checkpoint paths
CHECKPOINT_PATH=""
for (( i=0; i<NUM_RUNS; i++ )); do
    CP="${CHECKPOINT_DIR}/run_${i}/checkpoint_best.pt"
    if [[ $i -gt 0 ]]; then
        CHECKPOINT_PATH+=":"
    fi
    CHECKPOINT_PATH+="$CP"
done

TRANS_DIR="${OUTPUT_DIR}/ensemble_${NUM_RUNS}_models"
mkdir -p "$TRANS_DIR"

OUTPUT_FILE="${TRANS_DIR}/ensemble.out.interactive"

if [[ -f "$OUTPUT_FILE" ]]; then
    echo "Ensemble translation already exists: $OUTPUT_FILE"
    exit 0
fi

echo "Generating ensemble of ${NUM_RUNS} models → ${OUTPUT_FILE}"
fairseq-interactive \
    --path "$CHECKPOINT_PATH" "$DATA_BIN" \
    --input "$TEST_INPUT" \
    --beam 5 \
    --source-lang "$SOURCE_LANG" --target-lang "$TARGET_LANG" \
    --tokenizer moses \
    --bpe subword_nmt --bpe-codes "$BPE_CODES" \
    --remove-bpe | tee "$OUTPUT_FILE"
