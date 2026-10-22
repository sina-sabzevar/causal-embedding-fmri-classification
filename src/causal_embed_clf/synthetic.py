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
