"""Run per-region classification and write a results table and ROC figures.

Examples
--------
Synthetic data (no files needed):
    python scripts/run_classification.py --synthetic --out results/demo

Two group files, each (subjects, regions, dim):
    python scripts/run_classification.py --controls data/controls.npy \
        --patients data/patients.npy --out results/run1

Undersample controls to 55, repeated over 10 random subsets:
    python scripts/run_classification.py --controls data/controls.npy \
        --patients data/patients.npy --n-controls 55 --n-subsamples 10 --out results/run2
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from causal_embed_clf import (  # noqa: E402
    RESAMPLING_OPTIONS, EvalConfig, benjamini_hochberg, evaluate_all_regions,
    load_combined, load_groups, make_synthetic_embeddings, out_of_fold_scores,
    permutation_pvalue, subsample_controls,
)
from causal_embed_clf.viz import plot_roc  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("data (choose one)")
    src.add_argument("--controls", help="control embeddings .npy (subjects, regions, dim)")
    src.add_argument("--patients", help="patient embeddings .npy (subjects, regions, dim)")
    src.add_argument("--embeddings", help="all embeddings .npy, used with --labels")
    src.add_argument("--labels", help="0/1 labels .npy, used with --embeddings")
    src.add_argument("--synthetic", action="store_true", help="use synthetic demo data")

    p.add_argument("--n-controls", type=int, help="randomly keep this many controls")
    p.add_argument("--n-subsamples", type=int, default=1, help="number of random control subsets")
    p.add_argument("--k-features", type=int, default=30)
    p.add_argument("--resampling", choices=RESAMPLING_OPTIONS, default="smotetomek")
    p.add_argument("--kernel", default="linear", choices=["linear", "rbf"])
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=0.0, help="decision-function threshold")
    p.add_argument("--cv", choices=["loo", "repeated-kfold"], default="loo")
    p.add_argument("--permutation-top", type=int, default=0,
                   help="permutation-test the N best regions (BH-corrected)")
    p.add_argument("--n-permutations", type=int, default=200)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/run")
    return p.parse_args()


def load(args):
    if args.synthetic:
        return make_synthetic_embeddings(seed=args.seed)
    if args.controls and args.patients:
        return load_groups(args.controls, args.patients)
    if args.embeddings and args.labels:
        return load_combined(args.embeddings, args.labels)
    sys.exit("Give --controls and --patients, or --embeddings and --labels, or --synthetic")


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load(args)
    cfg = EvalConfig(k_features=args.k_features, resampling=args.resampling, kernel=args.kernel,
                     C=args.C, threshold=args.threshold, cv=args.cv, random_state=args.seed)

    seeds = range(args.seed, args.seed + args.n_subsamples) if args.n_controls else [None]
    tables = []
    for s in seeds:
        d = subsample_controls(data, args.n_controls, s) if s is not None else data
        print(f"[subset seed={s}] {int((d.y == 0).sum())} controls, {int((d.y == 1).sum())} patients, "
              f"{d.n_regions} regions")
        t = evaluate_all_regions(d.X, d.y, cfg, n_jobs=args.n_jobs)
        t["subset_seed"] = s
        tables.append(t)

    full = pd.concat(tables)
    full.to_csv(out / "regions_all_subsets.csv")
    metric_cols = [c for c in full.columns if c not in ("error", "subset_seed")]
    summary = full.groupby(level="region")[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.sort_values("auc_mean", ascending=False)

    errors = full.loc[full["error"].astype(str) != "", "error"] if "error" in full else pd.Series(dtype=str)
    if len(errors):
        print(f"WARNING: {len(errors)} region evaluations failed; see 'error' column")

    # Permutation test on the top regions, on the first subset only
    if args.permutation_top > 0:
        d = subsample_controls(data, args.n_controls, args.seed) if args.n_controls else data
        top = summary.index[: args.permutation_top]
        pvals = []
        for r in top:
            _, pv = permutation_pvalue(d.X[:, r, :], d.y, cfg, args.n_permutations, seed=args.seed)
            pvals.append(pv)
            print(f"  region {r}: permutation p = {pv:.4f}")
        summary.loc[top, "perm_p"] = pvals
        # Correct for every region that was eligible for selection, not just those tested
        all_p = np.ones(len(summary))
        all_p[: len(pvals)] = pvals
        summary["perm_p_bh"] = benjamini_hochberg(all_p)
        summary.loc[~summary.index.isin(top), "perm_p_bh"] = np.nan

    summary.to_csv(out / "regions_summary.csv")

    # One ROC per top region (never pooled across regions)
    d = subsample_controls(data, args.n_controls, args.seed) if args.n_controls else data
    for r in summary.index[:5]:
        scores = np.mean(out_of_fold_scores(d.X[:, r, :], d.y, cfg), axis=0)
        plot_roc(d.y, scores, f"Region {r}", out / f"roc_region_{r}.png")

    (out / "config.json").write_text(json.dumps({**asdict(cfg), **vars(args)}, indent=2, default=str))
    if args.n_subsamples <= 1 or not args.n_controls:
        summary = summary[[c for c in summary.columns if not c.endswith("_std")]]
        summary.columns = [c.removesuffix("_mean") for c in summary.columns]
        summary.to_csv(out / "regions_summary.csv")
    show = [c for c in summary.columns if c.split("_mean")[0] in
            ("auc", "balanced_accuracy", "sensitivity", "specificity") or c.startswith("perm")]
    print(summary[show].head(10).round(3).to_string())
    print(f"\nResults written to {out}/")


if __name__ == "__main__":
    main()
