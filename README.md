# Causal connectivity embeddings for fMRI classification

Per-region classification of patients versus controls from **node embeddings of directed (causal) brain connectivity graphs** estimated from resting-state fMRI.

This repository contains the classification and visualisation stage of:

> Sabzevar, S., Masoudnia, S., Araabi, B. N., NazemZadeh, M., & Tavakoli, H. (2024). Enhanced diagnosis of schizophrenia using integrated node embedding and causal discovery on imbalanced fMRI data. *10th International Conference on Signal Processing and Intelligent Systems (ICSPIS)*, IEEE, 234–239. https://doi.org/10.1109/ICSPIS65223.2024.10931071

## What it does

For each brain region, the node embedding of that region (one vector per subject) is used to classify patients against controls. The question is which regions' causal connectivity profiles carry diagnostic information, in small and class-imbalanced cohorts.

```
causal connectivity graph per subject   (estimated beforehand, e.g. rDCM or LiNGAM)
        -> node embeddings                (e.g. node2vec)   shape: subjects x regions x dim
        -> for each region:
              mRMR feature selection -> scaling -> resampling -> SVM
              evaluated by leave-one-out or repeated stratified k-fold CV
```

## Evaluation protocol

All steps that learn from data run **inside each cross-validation fold**, on the training subjects only:

- **Feature selection** (mRMR) is fitted per fold. Selecting features on the full dataset before cross-validation leaks the held-out subject's label into the model and inflates accuracy, often substantially in cohorts of tens of subjects.
- **Scaling** is fitted on the training fold only.
- **Resampling** (SMOTE, SVM-SMOTE, SMOTE-Tomek, or NearMiss followed by SVM-SMOTE and Tomek links) is applied to the training fold only. Held-out subjects are never resampled.
- **One ROC curve per region.** Scores from different regions' models are never pooled into a single curve.
- **Region selection is tested.** Picking the best regions out of many and reporting their accuracy is optimistic, so the best regions can be checked with a permutation test that re-runs the whole pipeline on shuffled labels, with Benjamini–Hochberg correction across all regions.
- **Control subsampling is repeated.** If controls are undersampled, the analysis runs over several random subsets and reports the spread, not a single arbitrary subset.

This is a cleaned re-implementation of the classification stage under the protocol above. Numbers obtained with it may differ from those reported in the paper.

## Not included

- fMRI preprocessing
- Causal discovery (rDCM, PC, GES, LiNGAM)
- Node embedding (node2vec)
- Any participant data

The code expects precomputed embedding arrays as input. A synthetic data generator is included so everything can be run and tested without patient data.

## Installation

```bash
git clone https://github.com/sina-sabzevar/causal-embedding-fmri-classification.git
cd causal-embedding-fmri-classification
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start (synthetic data)

```bash
python scripts/run_classification.py --synthetic --out results/demo
```

The synthetic data has 20 regions, of which regions 3 and 7 differ between groups. They should appear at the top of `results/demo/regions_summary.csv`.

## Running on your own embeddings

One array per group, each of shape `(subjects, regions, embedding_dim)`:

```bash
python scripts/run_classification.py \
    --controls data/controls.npy --patients data/patients.npy \
    --resampling smotetomek --cv loo --out results/run1
```

Or one array plus a 0/1 label vector: `--embeddings data/X.npy --labels data/y.npy`.

Useful options:

| Option | Meaning |
|---|---|
| `--resampling` | `none`, `smote`, `svmsmote`, `smotetomek`, `nearmiss-svmsmote-tomek` |
| `--cv` | `loo` or `repeated-kfold` (5 folds x 10 repeats) |
| `--k-features` | number of mRMR features per region (default 30) |
| `--kernel`, `--C` | SVM settings |
| `--threshold` | decision-function threshold (default 0) |
| `--n-controls`, `--n-subsamples` | undersample controls, repeated over several seeds |
| `--permutation-top N` | permutation-test the N best regions |
| `--n-jobs` | regions evaluated in parallel |

Outputs: `regions_summary.csv` (one row per region: AUC, balanced accuracy, sensitivity, specificity, precision, F1), `regions_all_subsets.csv`, one ROC figure per top region, and `config.json` recording every setting.

## Figures and BrainNet Viewer

```bash
python scripts/make_figures.py heatmap     --matrix data/rdcm.npy --subject 5 --out fig/heatmap.png
python scripts/make_figures.py connections --matrix data/rdcm.npy --regions 5 25 50 --out fig/connections.png
python scripts/make_figures.py brainnet    --nodes-csv data/top_regions.csv --out fig/regions.node
python scripts/make_figures.py resampling  --synthetic --region 3 --out fig/resampling.png
```

`--matrix` must be a square region-by-region connectivity matrix (or a stack of them), not an embedding array. The node CSV needs the columns `x,y,z,size,colour,label`.

## Tests

```bash
pytest
```

The tests check, among other things, that feature selection is fitted on n−1 subjects under leave-one-out, that every resampling option runs, and that an informative region is detected while a noise region is not.

## Citation

```bibtex
@inproceedings{sabzevar2024schizophrenia,
  author    = {Sabzevar, Sina and Masoudnia, Saeed and Araabi, Babak N. and NazemZadeh, MohammadReza and Tavakoli, Hosna},
  title     = {Enhanced Diagnosis of Schizophrenia Using Integrated Node Embedding and Causal Discovery on Imbalanced fMRI Data},
  booktitle = {2024 10th International Conference on Signal Processing and Intelligent Systems (ICSPIS)},
  pages     = {234--239},
  year      = {2024},
  publisher = {IEEE},
  doi       = {10.1109/ICSPIS65223.2024.10931071}
}
```

## License

MIT
