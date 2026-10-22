"""Leakage-free, per-region classification of patients vs controls.

For every brain region, a classifier is trained on that region's node
embedding. Everything that learns from data happens *inside* each
cross-validation fold:

    mRMR feature selection -> scaling -> resampling -> SVM

so no held-out subject ever influences feature selection, scaling or
resampling. Resampling is applied to the training fold only (imbalanced-learn
pipelines skip samplers at prediction time).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE, SVMSMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import NearMiss, TomekLinks
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .mrmr import MRMRSelector

RESAMPLING_OPTIONS = ("none", "smote", "svmsmote", "smotetomek", "nearmiss-svmsmote-tomek")


@dataclass
class EvalConfig:
    k_features: int = 30
    resampling: str = "smotetomek"
    kernel: str = "linear"
    C: float = 1.0
    class_weight: dict | str | None = "balanced"
    threshold: float = 0.0          # applied to the SVM decision function
    cv: str = "loo"                 # "loo" or "repeated-kfold"
    n_splits: int = 5
    n_repeats: int = 10
    random_state: int = 0
    extra: dict = field(default_factory=dict)


def _sampler_steps(name: str, random_state: int) -> list:
    if name == "none":
        return []
    if name == "smote":
        return [("smote", SMOTE(random_state=random_state))]
    if name == "svmsmote":
        return [("svmsmote", SVMSMOTE(random_state=random_state))]
    if name == "smotetomek":
        return [("smotetomek", SMOTETomek(random_state=random_state))]
    if name == "nearmiss-svmsmote-tomek":
        return [
            ("nearmiss", NearMiss(version=2, sampling_strategy=0.6)),
            ("svmsmote", SVMSMOTE(random_state=random_state)),
            ("tomek", TomekLinks()),
        ]
    raise ValueError(f"Unknown resampling '{name}'. Options: {RESAMPLING_OPTIONS}")


def build_pipeline(cfg: EvalConfig) -> Pipeline:
    """Feature selection, scaling, resampling and SVM as one fold-safe pipeline."""
    steps = [("select", MRMRSelector(k=cfg.k_features)), ("scale", StandardScaler())]
    steps += _sampler_steps(cfg.resampling, cfg.random_state)
    steps.append(("svm", SVC(kernel=cfg.kernel, C=cfg.C, class_weight=cfg.class_weight)))
    return Pipeline(steps)


def _splits(cfg: EvalConfig, y: np.ndarray):
    """Yield lists of (train, test) splits; one list per repeat."""
    if cfg.cv == "loo":
        yield list(LeaveOneOut().split(np.zeros(len(y))))
        return
    if cfg.cv == "repeated-kfold":
        rskf = RepeatedStratifiedKFold(
            n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, random_state=cfg.random_state
        )
        all_splits = list(rskf.split(np.zeros(len(y)), y))
        for r in range(cfg.n_repeats):
            yield all_splits[r * cfg.n_splits:(r + 1) * cfg.n_splits]
        return
    raise ValueError("cv must be 'loo' or 'repeated-kfold'")


def out_of_fold_scores(X: np.ndarray, y: np.ndarray, cfg: EvalConfig) -> list[np.ndarray]:
    """Decision-function scores for every subject, predicted while held out.

    Returns one score vector per CV repeat (a single vector for LOO).
    """
    base = build_pipeline(cfg)
    repeats = []
    for split_list in _splits(cfg, y):
        scores = np.full(len(y), np.nan)
        for train, test in split_list:
            model = clone(base)
            model.fit(X[train], y[train])
            scores[test] = model.decision_function(X[test])
        repeats.append(scores)
    return repeats


def metrics_from_scores(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": roc_auc_score(y, scores),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "accuracy": (tp + tn) / len(y),
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }


def evaluate_region(X_region: np.ndarray, y: np.ndarray, cfg: EvalConfig) -> dict:
    """Evaluate one region; metrics are averaged over CV repeats."""
    per_repeat = [metrics_from_scores(y, s, cfg.threshold) for s in out_of_fold_scores(X_region, y, cfg)]
    return {k: float(np.mean([m[k] for m in per_repeat])) for k in per_repeat[0]}


def _safe_evaluate(region: int, X: np.ndarray, y: np.ndarray, cfg: EvalConfig) -> dict:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            result = evaluate_region(X[:, region, :], y, cfg)
        result["error"] = ""
    except Exception as exc:  # recorded, not hidden: shows up in the results table
        result = {"error": f"{type(exc).__name__}: {exc}"}
    result["region"] = region
    return result


def evaluate_all_regions(X: np.ndarray, y: np.ndarray, cfg: EvalConfig, n_jobs: int = 1) -> pd.DataFrame:
    """Evaluate every region. Returns one row per region, best AUC first."""
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_safe_evaluate)(r, X, y, cfg) for r in range(X.shape[1])
    )
    df = pd.DataFrame(rows).set_index("region")
    if "auc" in df:
        df = df.sort_values("auc", ascending=False)
    return df


def permutation_pvalue(
    X_region: np.ndarray, y: np.ndarray, cfg: EvalConfig,
    n_permutations: int = 200, metric: str = "auc", seed: int = 0,
) -> tuple[float, float]:
    """Permutation test for one region: is the score better than chance?

    Labels are shuffled and the *whole* pipeline is re-run each time, so the
    null distribution includes feature selection and resampling.
    Returns (observed score, p-value).
    """
    rng = np.random.default_rng(seed)
    observed = evaluate_region(X_region, y, cfg)[metric]
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        null[i] = evaluate_region(X_region, rng.permutation(y), cfg)[metric]
    p = (1 + np.sum(null >= observed)) / (1 + n_permutations)
    return observed, float(p)


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """False-discovery-rate adjusted p-values (Benjamini-Hochberg)."""
    p = np.asarray(pvalues, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adjusted, 1.0)
    return out
