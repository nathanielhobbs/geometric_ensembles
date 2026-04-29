#!/usr/bin/env bash
# Evaluate an MT hypothesis file with standard metrics.
#
# Usage:
#   score.sh --hypothesis <file> --reference <file>
#            [--source <file>] [--metric bleu|ter|chrf|comet|comet-kiwi|bleu-rt]
#            [--lang-pair <xx-yy>]
#
# Options:
#   --hypothesis, -h   Path to hypothesis file (one translation per line)
#   --reference,  -r   Path to reference file
#   --source,     -s   Path to source file (required for comet and comet-kiwi)
#   --metric,     -m   bleu (default) | ter | chrf | comet | comet-kiwi | bleu-rt
#   --lang-pair,  -l   Language pair, e.g. de-en (used by sacrebleu tokenizer)
#   --help             Show this message
#
# Notes:
#   bleu, ter, chrf work out of the box.
#   comet and comet-kiwi download model weights on first use. To pre-cache:
#     python -c "from comet import download_model; download_model('Unbabel/wmt22-comet-da')"
#   bleu-rt (BLEURT) requires a separate checkpoint download — see:
#     https://github.com/google-research/bleurt

set -euo pipefail

HYPOTHESIS=""
REFERENCE=""
SOURCE=""
METRIC="bleu"
LANG_PAIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hypothesis|-h) HYPOTHESIS="$2"; shift 2 ;;
        --reference|-r)  REFERENCE="$2";  shift 2 ;;
        --source|-s)     SOURCE="$2";     shift 2 ;;
        --metric|-m)     METRIC="$2";     shift 2 ;;
        --lang-pair|-l)  LANG_PAIR="$2";  shift 2 ;;
        --help)
            sed -n '2,20p' "$0"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$HYPOTHESIS" || -z "$REFERENCE" ]]; then
    echo "Error: --hypothesis and --reference are required."
    echo "Run with --help for usage."
    exit 1
fi

case "$METRIC" in
    bleu)
        LANG_ARG=""
        [[ -n "$LANG_PAIR" ]] && LANG_ARG="-l $LANG_PAIR"
        sacrebleu "$REFERENCE" -i "$HYPOTHESIS" -m bleu $LANG_ARG
        ;;
    ter)
        sacrebleu "$REFERENCE" -i "$HYPOTHESIS" -m ter
        ;;
    chrf)
        sacrebleu "$REFERENCE" -i "$HYPOTHESIS" -m chrf
        ;;
    comet)
        if [[ -z "$SOURCE" ]]; then
            echo "Error: --source is required for COMET scoring."
            exit 1
        fi
        comet-score -s "$SOURCE" -t "$HYPOTHESIS" -r "$REFERENCE" --model Unbabel/wmt22-comet-da
        ;;
    comet-kiwi)
        if [[ -z "$SOURCE" ]]; then
            echo "Error: --source is required for COMET-Kiwi scoring."
            exit 1
        fi
        comet-score -s "$SOURCE" -t "$HYPOTHESIS" --model Unbabel/wmt22-cometkiwi-da
        ;;
    bleu-rt)
        if ! command -v bleurt &>/dev/null; then
            echo "Error: bleurt not found. See https://github.com/google-research/bleurt"
            exit 1
        fi
        bleurt score -candidate_file "$HYPOTHESIS" -reference_file "$REFERENCE"
        ;;
    *)
        echo "Unknown metric: $METRIC"
        echo "Supported: bleu, ter, chrf, comet, comet-kiwi, bleu-rt"
        exit 1
        ;;
esac
