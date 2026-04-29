#!/usr/bin/env bash
# Binarize preprocessed text data for fairseq training.
#
# Usage:
#   binarize.sh <config_file>
#
# Arguments:
#   config_file  Path to a dataset config file (see scripts/dataset_configs/).
#                Must define: PREPROCESSED_DIR, SOURCE_LANG, TARGET_LANG, DATA_BIN.
#
# Example:
#   binarize.sh scripts/dataset_configs/wmt23_cs_uk.cfg

set -euo pipefail

if [[ $# -lt 1 || "$1" == "--help" ]]; then
    sed -n '2,14p' "$0"
    exit 0
fi

source "$1"

: "${PREPROCESSED_DIR:?config must define PREPROCESSED_DIR}"
: "${SOURCE_LANG:?config must define SOURCE_LANG}"
: "${TARGET_LANG:?config must define TARGET_LANG}"
: "${DATA_BIN:?config must define DATA_BIN}"

fairseq-preprocess \
    --source-lang "$SOURCE_LANG" \
    --target-lang "$TARGET_LANG" \
    --trainpref "$PREPROCESSED_DIR/train" \
    --validpref "$PREPROCESSED_DIR/valid" \
    --destdir "$DATA_BIN" \
    --workers 20
