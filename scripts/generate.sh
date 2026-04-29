#!/usr/bin/env bash
# Translate a test set with a single fairseq checkpoint.
#
# Usage:
#   generate.sh <config_file> [run_num]
#
# Arguments:
#   config_file  Path to a dataset config file (see scripts/dataset_configs/).
#                Must define: DATA_BIN, BPE_CODES, TEST_INPUT, CHECKPOINT_DIR,
#                             OUTPUT_DIR, SOURCE_LANG, TARGET_LANG.
#   run_num      Integer run index matching the checkpoint to use (default: 0).
#
# The hypothesis file is written to OUTPUT_DIR/run_<run_num>/gen.out.nbest.interactive.
# To score the output, use scripts/score.sh.
#
# Example:
#   generate.sh scripts/dataset_configs/wmt23_cs_uk.cfg 3

set -euo pipefail

if [[ $# -lt 1 || "$1" == "--help" ]]; then
    sed -n '2,18p' "$0"
    exit 0
fi

source "$1"
RUN_NUM="${2:-0}"

: "${DATA_BIN:?config must define DATA_BIN}"
: "${BPE_CODES:?config must define BPE_CODES}"
: "${TEST_INPUT:?config must define TEST_INPUT}"
: "${CHECKPOINT_DIR:?config must define CHECKPOINT_DIR}"
: "${OUTPUT_DIR:?config must define OUTPUT_DIR}"
: "${SOURCE_LANG:?config must define SOURCE_LANG}"
: "${TARGET_LANG:?config must define TARGET_LANG}"

TRANS_DIR="${OUTPUT_DIR}/run_${RUN_NUM}"
mkdir -p "$TRANS_DIR"

CHECKPOINT="${CHECKPOINT_DIR}/run_${RUN_NUM}/checkpoint_best.pt"
OUTPUT_FILE="${TRANS_DIR}/gen.out.nbest.interactive"

if [[ -f "$OUTPUT_FILE" ]]; then
    echo "Translation already exists: $OUTPUT_FILE"
    exit 0
fi

echo "Translating run ${RUN_NUM} → ${OUTPUT_FILE}"
fairseq-interactive \
    --path "$CHECKPOINT" "$DATA_BIN" \
    --input "$TEST_INPUT" \
    --beam 5 --nbest 5 \
    --source-lang "$SOURCE_LANG" --target-lang "$TARGET_LANG" \
    --tokenizer moses \
    --bpe subword_nmt --bpe-codes "$BPE_CODES" \
    --remove-bpe | tee "$OUTPUT_FILE"
