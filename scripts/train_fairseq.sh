#!/usr/bin/env bash
# Train a fairseq Transformer model for one run.
#
# Usage:
#   train_fairseq.sh <config_file> [run_num]
#
# Arguments:
#   config_file  Path to a dataset config file (see scripts/dataset_configs/).
#                Must define: DATA_BIN, CHECKPOINT_DIR, ARCH, MAX_TOKENS,
#                             LR, WARMUP, DROPOUT, ENCODER_LAYERS, DECODER_LAYERS.
#   run_num      Integer run index used as the random seed (default: 0).
#
# Example:
#   train_fairseq.sh scripts/dataset_configs/wmt23_cs_uk.cfg 3

set -euo pipefail

if [[ $# -lt 1 || "$1" == "--help" ]]; then
    sed -n '2,16p' "$0"
    exit 0
fi

source "$1"
RUN_NUM="${2:-0}"

: "${DATA_BIN:?config must define DATA_BIN}"
: "${CHECKPOINT_DIR:?config must define CHECKPOINT_DIR}"
: "${ARCH:?config must define ARCH}"
: "${MAX_TOKENS:?config must define MAX_TOKENS}"
: "${LR:?config must define LR}"
: "${WARMUP:?config must define WARMUP}"
: "${DROPOUT:?config must define DROPOUT}"
: "${ENCODER_LAYERS:?config must define ENCODER_LAYERS}"
: "${DECODER_LAYERS:?config must define DECODER_LAYERS}"

SAVE_DIR="${CHECKPOINT_DIR}/run_${RUN_NUM}"
mkdir -p "$SAVE_DIR"

fairseq-train "$DATA_BIN" \
    --arch "$ARCH" \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr "$LR" --lr-scheduler inverse_sqrt \
    --warmup-updates "$WARMUP" --warmup-init-lr '1e-07' \
    --dropout "$DROPOUT" --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens "$MAX_TOKENS" --update-freq 1 \
    --encoder-layers "$ENCODER_LAYERS" --decoder-layers "$DECODER_LAYERS" \
    --encoder-embed-dim 1024 --decoder-embed-dim 1024 \
    --encoder-ffn-embed-dim 4096 --decoder-ffn-embed-dim 4096 \
    --encoder-attention-heads 16 --decoder-attention-heads 16 \
    --save-dir "$SAVE_DIR" --seed "$RUN_NUM"
