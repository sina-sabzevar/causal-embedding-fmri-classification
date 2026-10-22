"""Synthetic node-embedding data, so the pipeline can be run without patient data."""

from __future__ import annotations

import numpy as np

from .data import Dataset


def make_synthetic_embeddings(n_controls: int = 60, n_patients: int = 20, n_regions: int = 20,
                              dim: int = 64, informative_regions: tuple[int, ...] = (3, 7),
                              effect: float = 0.8, seed: int = 0) -> Dataset:
    """Random embeddings where only ``informative_regions`` differ between groups."""
    rng = np.random.default_rng(seed)
    n = n_controls + n_patients
    X = rng.normal(size=(n, n_regions, dim))
    y = np.r_[np.zeros(n_controls, int), np.ones(n_patients, int)]
    for r in (r for r in informative_regions if r < n_regions):
        X[y == 1, r, :8] += effect      # shift a few embedding dimensions in patients
    return Dataset(X=X, y=y)


def make_synthetic_timeseries(n_subjects: int = 10, n_regions: int = 8, n_timepoints: int = 200,
                              extra_edges: tuple[tuple[int, int], ...] = (),
                              strength: float = 0.8, seed: int = 0) -> list[np.ndarray]:
    """Time series from a linear non-Gaussian structural model on a fixed chain.

    Every subject shares the chain 0 -> 1 -> ... -> n_regions-1. ``extra_edges``
    adds further (source, target) edges with source < target, which can be used
    to make a patient group differ from controls. Noise is uniform (non-Gaussian),
    so LiNGAM's assumptions hold.
    """
    rng = np.random.default_rng(seed)
    B = np.zeros((n_regions, n_regions))          # B[i, j]: influence i -> j
    for i in range(n_regions - 1):
        B[i, i + 1] = strength
    for i, j in extra_edges:
        if not i < j:
            raise ValueError("extra edges must go from a lower to a higher index (acyclic)")
        B[i, j] = strength
    out = []
    for _ in range(n_subjects):
        X = np.zeros((n_timepoints, n_regions))
        noise = rng.uniform(-1, 1, size=(n_timepoints, n_regions))
        for j in range(n_regions):              # topological order is 0..n-1
            X[:, j] = X @ B[:, j] + noise[:, j]
        out.append(X)
    return out
