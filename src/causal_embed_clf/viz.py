"""Figures and export helpers: ROC curves, connectivity plots, BrainNet files."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath
from sklearn.metrics import roc_curve, auc


def plot_roc(y: np.ndarray, scores: np.ndarray, title: str, out_path: str | Path) -> float:
    """ROC curve for ONE model (one region). Do not pool scores across regions."""
    fpr, tpr, _ = roc_curve(y, scores)
    area = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {area:.2f}")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="grey")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate",
           ylabel="True positive rate", title=title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return float(area)


def plot_connectivity_heatmap(matrix: np.ndarray, out_path: str | Path,
                              highlight: list[int] | None = None,
                              title: str = "Directed connectivity") -> None:
    """Heatmap of a square region x region connectivity matrix."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square matrix, got {matrix.shape}")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", interpolation="none", aspect="auto")
    fig.colorbar(im, ax=ax, label="Connection strength")
    if highlight:
        ax.set_xticks(highlight)
        ax.set_yticks(highlight)
    ax.set(title=title, xlabel="Target region", ylabel="Source region")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_region_connections(matrix: np.ndarray, selected: list[int], out_path: str | Path,
                            threshold: float = 0.1, labels: list[str] | None = None) -> None:
    """Circular plot of connections from ``selected`` regions to all others.

    ``matrix`` must be a square region x region connectivity matrix
    (not a node-embedding array).
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square matrix, got {matrix.shape}")
    n = matrix.shape[0]
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xy = np.c_[np.cos(theta), np.sin(theta)]
    vmax = np.abs(matrix).max() or 1.0

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(*xy.T, s=8, color="lightgrey", zorder=1)
    cmap = plt.get_cmap("tab10")
    for i, src in enumerate(selected):
        for dst in range(n):
            w = matrix[src, dst]
            if dst == src or abs(w) <= threshold:
                continue
            path = MplPath([xy[src], (0, 0), xy[dst]], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
            ax.add_patch(FancyArrowPatch(path=path, color=cmap(i % 10),
                                         lw=0.5 + 3 * abs(w) / vmax, alpha=0.6, arrowstyle="-|>",
                                         mutation_scale=6))
        ax.scatter(*xy[src], s=120, color=cmap(i % 10), zorder=3)
        name = labels[src] if labels else str(src)
        ax.text(*(xy[src] * 1.12), name, ha="center", va="center", fontsize=8)
    ax.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), aspect="equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_brainnet_nodes(nodes: list[tuple[float, float, float, float, float, str]],
                         out_path: str | Path) -> None:
    """Write a BrainNet Viewer .node file: x y z size colour label per line."""
    with open(out_path, "w") as f:
        for x, y, z, size, colour, label in nodes:
            f.write(f"{x} {y} {z} {size} {colour} {label.strip()}\n")


def keep_rows_for_brainnet(matrix: np.ndarray, rows: list[int], out_path: str | Path) -> np.ndarray:
    """Zero every row not in ``rows`` and save the transposed matrix as .mat for BrainNet."""
    from scipy.io import savemat

    m = np.array(matrix, dtype=float, copy=True)   # never modify the caller's array
    mask = np.ones(m.shape[0], bool)
    mask[rows] = False
    m[mask, :] = 0.0
    savemat(out_path, {"matrix": m.T})
    return m


def plot_resampling_illustration(X2d: np.ndarray, y: np.ndarray, out_path: str | Path,
                                 random_state: int = 0,
                                 class_names: tuple[str, str] = ("Control", "Patient")) -> None:
    """Illustrate NearMiss -> SVM-SMOTE -> Tomek links on 2-D data.

    For figures only. ``X2d`` should be a 2-D projection of TRAINING data.
    The classifier itself resamples in the full feature space inside each fold.
    """
    from imblearn.over_sampling import SVMSMOTE
    from imblearn.under_sampling import NearMiss, TomekLinks

    y = np.asarray(y).ravel()
    X_nm, y_nm = NearMiss(version=1, sampling_strategy=0.5).fit_resample(X2d, y)
    X_sm, y_sm = SVMSMOTE(random_state=random_state).fit_resample(X_nm, y_nm)
    synthetic = np.zeros(len(y_sm), bool)
    synthetic[len(y_nm):] = True
    tomek = TomekLinks(sampling_strategy="majority")
    X_tl, y_tl = tomek.fit_resample(X_sm, y_sm)
    synthetic_tl = synthetic[tomek.sample_indices_]   # which kept points were synthetic

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    panels = [
        ("Original training data", X2d, y, np.zeros(len(y), bool)),
        ("After NearMiss", X_nm, y_nm, np.zeros(len(y_nm), bool)),
        ("After SVM-SMOTE", X_sm, y_sm, synthetic),
        ("After Tomek links", X_tl, y_tl, synthetic_tl),
    ]
    for ax, (title, Xp, yp, syn) in zip(axes.ravel(), panels):
        ax.scatter(*Xp[(yp == 0)].T, c="tab:blue", edgecolor="k", s=25, label=class_names[0])
        ax.scatter(*Xp[(yp == 1) & ~syn].T, c="tab:red", edgecolor="k", s=25, label=class_names[1])
        if syn.any():
            ax.scatter(*Xp[syn].T, c="tab:orange", edgecolor="k", s=25,
                       label=f"{class_names[1]} (synthetic)")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
