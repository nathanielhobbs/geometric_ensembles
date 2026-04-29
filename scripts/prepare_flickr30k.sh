#!/usr/bin/env bash
# Prepare Flickr30K / Multi30K de→en data for fairseq training.
#
# Usage:
#   prepare_flickr30k.sh
#
# Prerequisites:
#   - Raw Multi30K train/validation data (de+en files in current directory)
#   - Moses tokenizer (set SCRIPTS below)
#   - subword-nmt BPE tools (set BPEROOT below)
#   - Flickr30K test files (test_2017_flickr.{en,de}.lc.norm.tok)
#
# Data source: https://github.com/multi30k/dataset
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
    sed -n '2,17p' "$0"
    exit 0
fi

# Set these paths before running
SCRIPTS=mosesdecoder/scripts
BPEROOT=subword-nmt/subword_nmt
TEST_EN=test_2017_flickr.en.lc.norm.tok
TEST_DE=test_2017_flickr.de.lc.norm.tok

TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPE_TOKENS=10000

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS to point to Moses scripts (or run: git clone https://github.com/moses-smt/mosesdecoder.git)"
    exit 1
fi

src=de
tgt=en
prep=data.tokenized.de-en
tmp=$prep/tmp

mkdir -p $tmp $prep

for split in train val; do
    echo "Tokenizing ${split} en..."
    perl $TOKENIZER -threads 8 -l en < ${split}.en > $tmp/${split}.moses.tok.en
    echo "Tokenizing ${split} de..."
    perl $TOKENIZER -threads 8 -l de < ${split}.de > $tmp/${split}.moses.tok.de

    perl $LC < $tmp/${split}.moses.tok.en > $tmp/${split}.moses.tok.lc.en
    perl $LC < $tmp/${split}.moses.tok.de > $tmp/${split}.moses.tok.lc.de

    cp $tmp/${split}.moses.tok.lc.en $tmp/${split}.tok.en
    cp $tmp/${split}.moses.tok.de    $tmp/${split}.tok.de
    perl $CLEAN $tmp/${split}.tok $src $tgt $tmp/${split}.clean 1 150
done

TRAIN=$tmp/train.de-en
BPE_CODE=$prep/code.${BPE_TOKENS}.bpe
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.clean.$l >> $TRAIN
done

echo "Learning BPE with ${BPE_TOKENS} operations..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $tmp/train.clean.en > $prep/codes.${BPE_TOKENS}.bpe.en
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $tmp/train.clean.de > $prep/codes.${BPE_TOKENS}.bpe.de

VOCAB_EN=$prep/vocab.${BPE_TOKENS}.bpe.en
VOCAB_DE=$prep/vocab.${BPE_TOKENS}.bpe.de

python $BPEROOT/apply_bpe.py -c $prep/codes.${BPE_TOKENS}.bpe.en < $tmp/train.clean.en > $prep/train.${BPE_TOKENS}.bpe.en
python $BPEROOT/apply_bpe.py -c $prep/codes.${BPE_TOKENS}.bpe.de < $tmp/train.clean.de > $prep/train.${BPE_TOKENS}.bpe.de
cat $prep/train.${BPE_TOKENS}.bpe.en | python $BPEROOT/get_vocab.py > $VOCAB_EN
cat $prep/train.${BPE_TOKENS}.bpe.de | python $BPEROOT/get_vocab.py > $VOCAB_DE

for split in val; do
    python $BPEROOT/apply_bpe.py -c $prep/codes.${BPE_TOKENS}.bpe.en --vocabulary $VOCAB_EN < $tmp/${split}.clean.en > $prep/${split}.${BPE_TOKENS}.bpe.en
    python $BPEROOT/apply_bpe.py -c $prep/codes.${BPE_TOKENS}.bpe.de --vocabulary $VOCAB_DE < $tmp/${split}.clean.de > $prep/${split}.${BPE_TOKENS}.bpe.de
done

echo "Applying BPE to test set..."
python $BPEROOT/apply_bpe.py -c $prep/codes.${BPE_TOKENS}.bpe.en --vocabulary $VOCAB_EN < "$TEST_EN" > $prep/test.${BPE_TOKENS}.bpe.en
python $BPEROOT/apply_bpe.py -c $prep/codes.${BPE_TOKENS}.bpe.de --vocabulary $VOCAB_DE < "$TEST_DE" > $prep/test.${BPE_TOKENS}.bpe.de

echo "Done. Next step: run scripts/binarize.sh to create fairseq binary data."
