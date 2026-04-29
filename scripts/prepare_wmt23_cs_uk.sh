#!/usr/bin/env bash
# Prepare WMT23 cs→uk data for fairseq training.
#
# Usage:
#   prepare_wmt23_cs_uk.sh [--icml17]
#
# Options:
#   --icml17   Use the WMT14-compatible corpus variant (sets output to wmt14_en_de/).
#              Omit for the default WMT23 cs-uk setup (output: wmt23_cs_uk_russ_tok/).
#
# Expects train.cs and train.uk files in the current directory (from WMT23 training data).
# Tokenizes using Moses with Russian-language tokenizer for Ukrainian text, learns BPE,
# and applies it to produce train/valid splits ready for fairseq binarization.
#
# Data source: https://www2.statmt.org/wmt23/translation-task.html
# Prerequisites: Moses tokenizer and subword-nmt (cloned automatically).
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
    sed -n '2,18p' "$0"
    exit 0
fi

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=40000

if [ "${1:-}" == "--icml17" ]; then
    OUTDIR=wmt14_en_de
else
    OUTDIR=wmt23_cs_uk_russ_tok
fi

if [ ! -d "$SCRIPTS" ]; then
    echo "Moses scripts not found. Clone from https://github.com/moses-smt/mosesdecoder.git"
    exit 1
fi

src=cs
tgt=uk
lang=cs-uk
prep=$OUTDIR
tmp=$prep/tmp

mkdir -p $tmp $prep

CORPORA=("train")

echo "Pre-processing train data..."
for l in $src $tgt; do
    # Ukrainian text is tokenized with the Russian Moses tokenizer
    l2=$( [[ "$l" == "uk" ]] && echo "ru" || echo "cs" )
    l1=$( [[ "$l" == "uk" ]] && echo "uk" || echo "cs" )
    rm -f $tmp/train.tags.$lang.tok.$l1
    for f in "${CORPORA[@]}"; do
        cat $f.$l1 | \
            perl $NORM_PUNC $l2 | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 8 -a -l $l2 >> $tmp/train.tags.$lang.tok.$l1
    done
done

echo "Splitting train and valid (1 in every 100 lines → valid)..."
for l in $src $tgt; do
    l1=$( [[ "$l" == "uk" ]] && echo "uk" || echo "cs" )
    awk '{if (NR%100 == 0) print $0; }' $tmp/train.tags.$lang.tok.$l1 > $tmp/valid.$l
    awk '{if (NR%100 != 0) print $0; }' $tmp/train.tags.$lang.tok.$l1 > $tmp/train.$l
done

TRAIN=$tmp/train.cs-uk
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do cat $tmp/train.$l >> $TRAIN; done

echo "Learning BPE with ${BPE_TOKENS} operations..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "Applying BPE to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $tmp/bpe.$f
    done
done

perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250

echo "Done. Next step: run scripts/binarize.sh to create fairseq binary data."
