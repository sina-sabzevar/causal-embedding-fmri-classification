# Causal connectivity embeddings for fMRI classification

Finding brain regions whose **directed (causal) connectivity** distinguishes patients from controls, from resting-state fMRI, in small and class-imbalanced cohorts.

This repository is a cleaned re-implementation of the pipeline in:

> Sabzevar, S., Masoudnia, S., Araabi, B. N., NazemZadeh, M., & Tavakoli, H. (2024). Enhanced diagnosis of schizophrenia using integrated node embedding and causal discovery on imbalanced fMRI data. *10th International Conference on Signal Processing and Intelligent Systems (ICSPIS)*, IEEE, 234–239. https://doi.org/10.1109/ICSPIS65223.2024.10931071

Numbers obtained with this code may differ from those reported in the paper; see [Evaluation protocol](#evaluation-protocol).

## Pipeline

```
regional fMRI time series            one (timepoints x regions) array per subject
   |
   |  1. causal discovery            PC, GES or DirectLiNGAM   -> causal.py
   |                                 (or import rDCM matrices from MATLAB)
   v
directed connectivity matrix         one (regions x regions) matrix per subject
   |
   |  2. region features             node2vec (p=2, q=0.5)     -> embedding.py
   |                                 or connectivity profile
   v
region features                      (subjects x regions x dim)
   |
   |  3. per-region classification   inside every CV fold:     -> evaluate.py
   |                                 mRMR -> scaling -> resampling -> SVM
   v
AUC, balanced accuracy, sensitivity, specificity per region, with permutation tests
```

| Module | What it does |
|---|---|
| `causal.py` | PC (Fisher-z), GES (BIC) and DirectLiNGAM via causal-learn. Convention: `A[i, j]` is the influence of region *i* on region *j*. PC/GES edges whose direction is undetermined are kept in both directions; edge weights are OLS coefficients or binary. Loads rDCM or other matrices from `.npy`/`.mat`. |
| `embedding.py` | node2vec on directed, weighted graphs with [PecanPy](https://github.com/krishnanlab/PecanPy) (the implementation used in the paper), plus a short pure-Python version of the same algorithm. Also connectivity-profile features. |
| `resampling.py` | SMOTE, SVM-SMOTE, SMOTE-Tomek, and NearMiss → SVM-SMOTE → Tomek links. Used only inside training folds. |
| `mrmr.py` | mRMR feature selection as a scikit-learn transformer, so it can be fitted per fold. |
| `evaluate.py` | Per-region cross-validated evaluation, permutation tests, Benjamini–Hochberg correction. |
| `viz.py` | ROC curves, connectivity heatmaps, circular connection plots, BrainNet Viewer exports. |

## Evaluation protocol

All steps that learn from data run **inside each cross-validation fold**, on the training subjects only:

- **Feature selection** (mRMR) is fitted per fold. Selecting features on the full dataset before cross-validation leaks the held-out subject's label into the model and inflates accuracy, often substantially in cohorts of tens of subjects.
- **Scaling** is fitted on the training fold only.
- **Resampling** is applied to the training fold only. Held-out subjects are never resampled.
- **One ROC curve per region.** Scores from different regions' models are never pooled.
- **Region selection is tested.** The best regions can be checked with a permutation test that re-runs the whole pipeline on shuffled labels, with Benjamini–Hochberg correction across all regions.
- **Control subsampling is repeated.** If controls are undersampled, results are reported over several random subsets.

## Using PecanPy: negative edges and reproducibility

Causal graphs contain negative (inhibitory) edges, and PecanPy never walks along a negative edge. If a signed matrix is passed to it directly, those edges are silently ignored. By default the walks here therefore use absolute edge strengths (`--edge-weights abs`). `--edge-weights positive` keeps only positive edges, which reproduces what PecanPy does with signed input.

PecanPy also generates walks on several threads, and with more than one thread the same `random_state` gives a different embedding on every run. Walks are generated on one thread here, which makes results reproducible; subjects are processed in parallel instead (`--n-jobs`).

## A limitation of node2vec features

node2vec is fitted separately for each subject, and each subject's embedding space is only defined up to an arbitrary rotation, so the same coordinate does not necessarily mean the same thing in two subjects. Using one random seed for all subjects (the default here) makes the spaces partly comparable, not guaranteed to be.

`scripts/demo_end_to_end.py` shows the consequence on synthetic data where patients differ by a single edge, 2 → 4. Connectivity-profile features give AUC 1.00 to regions 2–6, the regions on and downstream of that edge. node2vec features (PecanPy, one seed for all subjects) also put regions 3 and 2 near the top, but regions 15, 12 and 13, which are unrelated to the edge, score 0.97, 0.88 and 0.87. When each subject's embedding was trained with a different seed, the group difference disappeared. Region-level conclusions from node2vec features should therefore be checked against `--features profile`, whose coordinates are comparable across subjects by construction.

## Not included

- fMRI preprocessing and parcellation
- rDCM estimation, which runs in MATLAB through the TAPAS toolbox. Export its matrices and load them with `--matrices`; pass `--transpose` if `A[i, j]` means *j → i*.
- Any participant data. `.gitignore` blocks `.npy`, `.mat` and NIfTI files.

## Installation

```bash
git clone https://github.com/sina-sabzevar/causal-embedding-fmri-classification.git
cd causal-embedding-fmri-classification
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

Whole pipeline on synthetic data, a few minutes (PecanPy compiles its walk function once per subject):

```bash
python scripts/demo_end_to_end.py
```

Classification stage alone on synthetic embeddings (regions 3 and 7 are the informative ones):

```bash
python scripts/run_classification.py --synthetic --out results/demo
```

## Running on your own data

**1. Build graphs and features, once per group.**

```bash
# time series as one array (subjects, timepoints, regions)
python scripts/build_embeddings.py --timeseries data/controls_ts.npy --method lingam --out data/controls
python scripts/build_embeddings.py --timeseries data/patients_ts.npy --method lingam --out data/patients

# or a folder with one (timepoints, regions) .npy per subject
python scripts/build_embeddings.py --timeseries-dir data/controls_ts/ --method pc --out data/controls

# or precomputed matrices, e.g. rDCM from MATLAB
python scripts/build_embeddings.py --matrices data/rdcm_controls.mat --transpose --out data/controls
```

Each run writes `<out>_graphs.npy`, `<out>_embeddings.npy` and `<out>_settings.json`. Options: `--method pc|ges|lingam`, `--features node2vec|profile`, `--backend pecanpy|builtin`, `--edge-weights abs|positive`, `--dim`, `--p`, `--q`, `--walk-length`, `--walks-per-node`, `--seed`, `--n-jobs`.

**2. Classify.**

```bash
python scripts/run_classification.py \
    --controls data/controls_embeddings.npy --patients data/patients_embeddings.npy \
    --resampling smotetomek --cv loo --permutation-top 10 --out results/run1
```

| Option | Meaning |
|---|---|
| `--resampling` | `none`, `smote`, `svmsmote`, `smotetomek`, `nearmiss-svmsmote-tomek` |
| `--cv` | `loo` or `repeated-kfold` (5 folds x 10 repeats) |
| `--k-features` | mRMR features per region (default 30) |
| `--kernel`, `--C`, `--threshold` | SVM settings; threshold applies to the decision function |
| `--n-controls`, `--n-subsamples` | undersample controls, repeated over several seeds |
| `--permutation-top N` | permutation-test the N best regions |

Outputs: `regions_summary.csv` (one row per region), `regions_all_subsets.csv`, one ROC figure per top region, and `config.json`.

**3. Figures and BrainNet Viewer.**

```bash
python scripts/make_figures.py heatmap     --matrix data/controls_graphs.npy --subject 0 --out fig/heatmap.png
python scripts/make_figures.py connections --matrix data/controls_graphs.npy --regions 5 25 50 --out fig/connections.png
python scripts/make_figures.py brainnet    --nodes-csv data/top_regions.csv --out fig/regions.node
python scripts/make_figures.py resampling  --synthetic --region 3 --out fig/resampling.png
```

The node CSV needs the columns `x,y,z,size,colour,label`.

## Tests

```bash
pytest
```

The tests check that LiNGAM recovers edge directions on non-Gaussian data, that PC and GES recover the skeleton, causal-learn's graph encoding is converted correctly, both node2vec backends are deterministic and separate two unconnected groups of regions, negative edges are handled as documented, feature selection is fitted on n−1 subjects under leave-one-out, every resampling option runs, and an informative region is detected while a noise region is not.

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
