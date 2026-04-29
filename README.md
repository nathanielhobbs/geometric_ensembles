# Geometric Ensembles for Machine Translation

Code for the paper:

> **Geometric Ensembles for Machine Translation**
> *IEEE Transactions on Big Data*. DOI: `TODO — fill in before publishing`

---

## Overview

Geometric ensembles is a **black-box** method for combining outputs from multiple MT systems without access to model weights or decoder internals. Given N independently trained models, the ensemble selects, for each source sentence, the hypothesis that is most *central* among all model outputs — the discrete analog of the geometric median.

Centrality is measured using **word-level edit distance** (primary) or **pairwise BLEU** (alternative). The selected hypothesis minimises the mean pairwise distance to all other model outputs, consistently outperforming single-model baselines and matching or exceeding white-box ensembles that require decoder access. An **XGBoost mixture-of-experts** component learns per-sentence features from the distance matrices to predict which system to select.

Evaluation metrics supported: BLEU, TER, CHRF, BLEU-RT, COMET, COMET-Kiwi.

---

## Repository Structure

```
geometric_ensembles/
├── main.py                              # Pipeline entry point
├── distances.py                         # Pairwise distances: edit, BLEU, BERT, COMET, Word2Vec
├── measures.py                          # f(g()) selection: argmin/argmax of mean/var/median/min/max
├── edit_distance.py                     # Word-level edit distance with optional substitution matrices
├── utils.py                             # Scoring (sacrebleu, TER, CHRF, COMET) and file I/O
├── xgboost_classifiers.py               # XGBoost mixture-of-experts system selector
├── configs/
│   ├── flickr30k/      # Flickr30K de→en image captioning, 25 models
│   ├── iwslt14/        # IWSLT14 de→en speech translation, 25 models
│   ├── wmt14_en_de/    # WMT14 en→de news translation, 25 models
│   └── wmt23_cs_uk/    # WMT23 cs→uk, 25 models
├── data/
│   ├── flickr30k/      # Source, reference, ensemble, 25 model hypotheses
│   ├── iwslt14/        # Source, reference, ensemble, 25 model hypotheses
│   ├── wmt14_en_de/    # Source, reference, ensemble, 25 model hypotheses
│   └── wmt23_cs_uk/    # Source, reference, ensemble, 25 model hypotheses
└── scripts/
    ├── score.sh                         # Evaluate hypotheses with any MT metric
    ├── prepare_flickr30k.sh             # Data preparation: Flickr30K
    ├── prepare_iwslt14.sh               # Data preparation: IWSLT14
    ├── prepare_wmt14_en_de.sh           # Data preparation: WMT14 en-de
    ├── prepare_wmt23_cs_uk.sh           # Data preparation: WMT23 cs-uk
    ├── binarize.sh                      # Fairseq binarization (any dataset)
    ├── train_fairseq.sh                 # Train a Transformer model (any dataset)
    ├── generate.sh                      # Translate with a single checkpoint
    ├── generate_ensemble.sh             # Translate with an ensemble of checkpoints
    └── dataset_configs/                 # Per-dataset config files for the fairseq scripts
        ├── flickr30k.cfg
        ├── iwslt14.cfg
        ├── wmt14_en_de.cfg
        └── wmt23_cs_uk.cfg
```

---

## Installation

```bash
conda env create -f environment.yml
conda activate geometric_ensembles
```

Core dependencies: `sacrebleu`, `unbabel-comet`, `editdistance`, `gensim`, `transformers`, `numpy`, `h5py`, `scikit-learn`, `xgboost`, PyTorch ≥ 2.0.

---

## Quick Start

### 1. Use the provided data or prepare your own

The `data/` directory contains the hypothesis files, source, and reference translations used in the paper (see [Reproducibility](#reproducibility) below). To run on your own data, prepare plain-text files with one translation per line, in the same sentence order as the reference.

### 2. Create a config

Copy an existing config directory as a starting point:

```bash
cp -r configs/wmt14_en_de configs/my_experiment
```

Edit `configs/my_experiment/config.py`:

```python
COMPUTE_TRAINING = False    # set True to evaluate on a training split as well
ENSEMBLE_INCLUDED = False   # set True to prepend a standard (white-box) ensemble as one system
NUM_PROCESSES = 4           # parallel workers for distance computation
base = "my_experiment"

results_path   = "results/my_experiment"
source_path    = "data/my_experiment/source.txt"
reference_path = "data/my_experiment/reference.txt"
log_file_path  = "logs/my_experiment.log"

model_hypothesis_paths = [
    "data/my_experiment/model_01.txt",
    "data/my_experiment/model_02.txt",
    # one path per system
]
```

### 3. Run

```bash
# Named config directory under configs/:
python main.py flickr30k
python main.py iwslt14
python main.py wmt14_en_de
python main.py wmt23_cs_uk
python main.py my_experiment
```

Results (per-metric scores, selected hypotheses) are written to `results_path`.

---

## Reproducibility

### What is included

The `data/` directory contains everything needed to reproduce the paper's results:

| Dataset | Source | Reference | Ensemble | Model hypotheses |
|---|---|---|---|---|
| Flickr30K | ✓ | ✓ | ✓ | 25 files |
| IWSLT14 de→en | ✓ | ✓ | ✓ | 25 files |
| WMT14 en→de | ✓ | ✓ | ✓ | 25 files |
| WMT23 cs→uk | ✓ | ✓ | ✓ | 25 files |

All hypothesis files are plain text (one translation per line). The ensemble file in each directory contains translations from the standard white-box ensemble baseline.

### What is not included

The following intermediate artefacts are **not** included due to size (10s–100s of GB):

- **Substitution matrices** for weighted edit distance
- **Precomputed pairwise edit distance matrices** (`.npy` files)
- **Precomputed pairwise BLEU distance matrices** (`.npy` files)
- **Precomputed COMET/BERT embedding distance matrices**

These are recomputed automatically on first run and cached to `cache/distances/`. Runtime for the first run depends on dataset size and available CPUs (`NUM_PROCESSES` in config).

---

## Scripts

All scripts accept `--help` for usage information.

### Scoring (`score.sh`)

Evaluate any hypothesis file against a reference using standard MT metrics:

```bash
scripts/score.sh --hypothesis data/wmt14_en_de/model_01.txt \
                 --reference  data/wmt14_en_de/reference.txt \
                 --metric bleu --lang-pair en-de

scripts/score.sh --hypothesis data/wmt23_cs_uk/model_01.txt \
                 --reference  data/wmt23_cs_uk/reference.txt \
                 --source     data/wmt23_cs_uk/source.txt \
                 --metric comet
```

Supported metrics: `bleu`, `ter`, `chrf`, `comet`, `comet-kiwi`, `bleu-rt`.

> **Note:** `bleu`, `ter`, and `chrf` work out of the box. `comet` and `comet-kiwi` download model weights on first use (run `python -c "from comet import download_model; download_model('Unbabel/wmt22-comet-da')"` in advance to cache them). `bleu-rt` (BLEURT) requires a separate checkpoint download — see the [BLEURT repository](https://github.com/google-research/bleurt) for setup instructions.

### Data preparation

Each dataset has a dedicated preparation script. Run from the directory where you want the preprocessed data to land:

```bash
scripts/prepare_flickr30k.sh          # Flickr30K de-en: tokenise, BPE, binarise
scripts/prepare_iwslt14.sh            # IWSLT14 de-en: download, tokenise, BPE
scripts/prepare_wmt14_en_de.sh        # WMT14 en-de: download, tokenise, BPE
scripts/prepare_wmt23_cs_uk.sh        # WMT23 cs-uk: tokenise, BPE
```

### Fairseq training and inference

`binarize.sh`, `train_fairseq.sh`, `generate.sh`, and `generate_ensemble.sh` are general-purpose fairseq scripts parameterised by a dataset config file. The WMT23 cs-uk config (`scripts/dataset_configs/wmt23_cs_uk.cfg`) is the fully worked example used in the paper; configs for the other datasets are provided as starting points.

```bash
# Binarise preprocessed text for fairseq
scripts/binarize.sh scripts/dataset_configs/wmt23_cs_uk.cfg

# Train a single model (run_num sets the random seed and checkpoint directory)
scripts/train_fairseq.sh scripts/dataset_configs/wmt23_cs_uk.cfg 0

# Translate with a single checkpoint
scripts/generate.sh scripts/dataset_configs/wmt23_cs_uk.cfg 0

# Translate with an ensemble of N checkpoints
scripts/generate_ensemble.sh scripts/dataset_configs/wmt23_cs_uk.cfg 19
```

To adapt to a different dataset, copy any `.cfg` file from `scripts/dataset_configs/`, update the fields, and pass the new config as the first argument.

---

## Datasets Used in the Paper

Hypothesis files are provided in `data/`. Raw training data must be obtained from the original sources:

| Dataset | Task | Source |
|---|---|---|
| Flickr30K / Multi30K | de→en image captioning | [multi30k/dataset](https://github.com/multi30k/dataset) |
| IWSLT14 | de→en speech | [IWSLT archive](https://wit3.fbk.eu/) |
| WMT14 | en→de news | [statmt.org/wmt14](https://www.statmt.org/wmt14/translation-task.html) |
| WMT23 | cs→uk news | [statmt.org/wmt23](https://www2.statmt.org/wmt23/translation-task.html) |

---

## Citation

```bibtex
@article{hobbs_geometric_ensembles,
  title   = {Geometric Ensembles for Machine Translation},
  author  = {TODO},
  journal = {IEEE Transactions on Big Data},
  year    = {TODO},
  doi     = {TODO}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
