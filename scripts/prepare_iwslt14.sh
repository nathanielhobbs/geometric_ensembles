#!/usr/bin/env bash
# Download and prepare IWSLT14 de→en data for fairseq training.
#
# Usage:
#   prepare_iwslt14.sh
#
# Downloads the IWSLT14 de→en archive from the fairseq CDN, tokenizes,
# cleans, and applies BPE. Output is written to iwslt14.tokenized.de-en/.
#
# Prerequisites: Moses tokenizer and subword-nmt (cloned automatically).
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
    sed -n '2,14p' "$0"
    exit 0
fi

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=10000

URL="http://dl.fbaipublicfiles.com/fairseq/data/iwslt14/de-en.tgz"
GZ=de-en.tgz

if [ ! -d "$SCRIPTS" ]; then
    echo "Moses scripts not found. Clone from https://github.com/moses-smt/mosesdecoder.git"
    exit 1
fi

src=de
tgt=en
lang=de-en
prep=iwslt14.tokenized.de-en
tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep

echo "Downloading data from ${URL}..."
cd $orig
wget "$URL"
[ -f $GZ ] || { echo "Download failed."; exit 1; }
tar zxvf $GZ
cd ..

echo "Pre-processing train data..."
for l in $src $tgt; do
    f=train.tags.$lang.$l
    tok=train.tags.$lang.tok.$l
    cat $orig/$lang/$f | \
        grep -v '<url>' | grep -v '<talkid>' | grep -v '<keywords>' | \
        sed -e 's/<title>//g' -e 's/<\/title>//g' \
            -e 's/<description>//g' -e 's/<\/description>//g' | \
        perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
done
perl $CLEAN -ratio 1.5 $tmp/train.tags.$lang.tok $src $tgt $tmp/train.tags.$lang.clean 1 175
for l in $src $tgt; do
    perl $LC < $tmp/train.tags.$lang.clean.$l > $tmp/train.tags.$lang.$l
done

echo "Pre-processing valid/test data..."
for l in $src $tgt; do
    for o in $(ls $orig/$lang/IWSLT14.TED*.$l.xml); do
        fname=${o##*/}
        f=$tmp/${fname%.*}
        grep '<seg id' $o | \
            sed -e 's/<seg id="[0-9]*">\s*//g' -e 's/\s*<\/seg>\s*//g' -e "s/\'/\'/g" | \
            perl $TOKENIZER -threads 8 -l $l | perl $LC > $f
    done
done

echo "Creating train/valid/test splits..."
for l in $src $tgt; do
    awk '{if (NR%23 == 0) print $0; }' $tmp/train.tags.de-en.$l > $tmp/valid.$l
    awk '{if (NR%23 != 0) print $0; }' $tmp/train.tags.de-en.$l > $tmp/train.$l
    cat $tmp/IWSLT14.TED.dev2010.de-en.$l \
        $tmp/IWSLT14.TEDX.dev2012.de-en.$l \
        $tmp/IWSLT14.TED.tst2010.de-en.$l \
        $tmp/IWSLT14.TED.tst2011.de-en.$l \
        $tmp/IWSLT14.TED.tst2012.de-en.$l > $tmp/test.$l
done

TRAIN=$tmp/train.en-de
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do cat $tmp/train.$l >> $TRAIN; done

echo "Learning BPE with ${BPE_TOKENS} operations..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "Applying BPE to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
    done
done

echo "Done. Next step: run scripts/binarize.sh to create fairseq binary data."
